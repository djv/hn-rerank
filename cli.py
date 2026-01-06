import argparse
import asyncio
import httpx
import numpy as np
import gc
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from api.main import get_best_stories, fetch_story_with_comments
from api import rerank
from api.config import get_username as get_saved_username, save_config

console = Console()


async def fetch_story_texts(client, story_ids):
    """Fetch text content for a list of story IDs."""
    if not story_ids:
        return []
    # Limit to top 50 to avoid too many requests
    ids_to_fetch = sorted(list(story_ids), reverse=True)[:50]
    tasks = [fetch_story_with_comments(client, int(i)) for i in ids_to_fetch]
    results = await asyncio.gather(*tasks)
    texts = []
    for story in results:
        if story and story.get("text_content"):
            texts.append(story["text_content"])
    return texts


async def main(args):
    username = args.username or get_saved_username()

    if not username:
        console.print(
            "[red]Error: Username not provided. Please provide a username once to save it.[/]"
        )
        console.print("Usage: uv run cli.py <username>")
        return

    # Save for next time if explicit
    if args.username:
        save_config("username", username)

    # Initialize Model
    rerank.init_model(args.model)

    from api.client import HNClient

    hn_client = HNClient()

    # Setup Progress Bar columns
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        # Define tasks
        task_scraping = progress.add_task(
            f"[cyan]Fetching user data for {username}...", total=None
        )
        task_fetch = progress.add_task(
            f"[cyan]Fetching Top {args.limit} stories...", total=None
        )

        async def run_scraping():
            # Check login status
            is_logged_in = await hn_client.check_session()
            if not is_logged_in:
                console.print(
                    "[yellow]Not logged in. Only public favorites will be used. Run --login to use upvoted/hidden stories.[/]"
                )

            # Fetch IDs
            fav_ids = await hn_client.fetch_favorites(username)
            upvoted_ids = set()
            hidden_ids = set()

            if is_logged_in:
                upvoted_ids = await hn_client.fetch_upvoted(username)
                hidden_ids = await hn_client.fetch_hidden(username)

            # Combine positive (Favs + Upvoted)
            positive_ids = fav_ids.union(upvoted_ids)
            negative_ids = hidden_ids

            progress.console.print(
                f"[dim]Found {len(fav_ids)} favorites, {len(upvoted_ids)} upvoted, {len(hidden_ids)} hidden.[/]"
            )

            # Fetch Texts (using a separate client for API calls)
            async with httpx.AsyncClient() as api_client:
                positive_texts = await fetch_story_texts(api_client, positive_ids)
                negative_texts = await fetch_story_texts(api_client, negative_ids)

            progress.update(task_scraping, completed=100)
            # Combine all user IDs for filtering
            all_user_ids = positive_ids.union(negative_ids)
            return positive_texts, negative_texts, all_user_ids

        async def run_fetching():
            res = await get_best_stories(limit=args.limit, days=args.days)
            progress.update(task_fetch, completed=100)
            return res

        # Run concurrently
        (pos_texts, neg_texts, user_story_ids), candidates_raw = await asyncio.gather(
            run_scraping(), run_fetching()
        )

        await hn_client.close()

        if not candidates_raw:
            console.print("[red]Failed to fetch candidate stories.[/]")
            return

        # Filter out stories already seen/interacted
        original_count = len(candidates_raw)
        candidates_raw = [c for c in candidates_raw if c["id"] not in user_story_ids]
        filtered_count = original_count - len(candidates_raw)

        if filtered_count > 0:
            progress.console.print(
                f"[dim]Filtered out {filtered_count} stories you already interacted with.[/]"
            )

        # 3. Rerank
        task_rerank = progress.add_task(
            "[cyan]Calculating semantic scores...", total=len(candidates_raw)
        )

        candidates_text = [
            s.get("text_content", s.get("title", "")) for s in candidates_raw
        ]

        # Embed Data
        # Positive
        pos_embeddings = await asyncio.to_thread(
            rerank.get_embeddings, pos_texts, is_query=True
        )
        # Negative
        neg_embeddings = await asyncio.to_thread(
            rerank.get_embeddings, neg_texts, is_query=True
        )

        # Batch embedding candidates
        batch_size = 4
        all_embeddings = []

        for i in range(0, len(candidates_text), batch_size):
            batch = candidates_text[i : i + batch_size]
            try:
                batch_embeddings = await asyncio.to_thread(rerank.get_embeddings, batch)
                all_embeddings.append(batch_embeddings)
                if (i // batch_size) % 5 == 0:
                    gc.collect()
            except Exception as e:
                console.print(f"[yellow]Skipping batch: {e}[/]")
                all_embeddings.append(np.zeros((len(batch), 768)))

            progress.advance(task_rerank, advance=len(batch))

        if all_embeddings:
            full_embeddings = np.vstack(all_embeddings)

            ranked_indices = rerank.rank_stories(
                candidates_raw,
                cand_embeddings=full_embeddings,
                positive_embeddings=pos_embeddings,
                negative_embeddings=neg_embeddings,
                diversity_lambda=args.diversity,
                hn_weight=0.2,
                neg_weight=0.5,
            )
        else:
            ranked_indices = []

    # 4. Print Results
    console.print(
        f"\n[bold green]Top {args.top} Personalized Stories for {username} (Last {args.days} Days)[/]\n"
    )

    shown_count = 0

    for idx, score, fav_idx in ranked_indices:
        if shown_count >= args.top:
            break
        shown_count += 1

        story = candidates_raw[idx]

        # Determine "Why" (matched positive)
        # We don't have direct mapping back to text easily in this simplified rank_stories
        # But we can try if fav_idx is valid
        # rank_stories returns (cand_idx, score, best_pos_idx)

        reason = ""
        if fav_idx >= 0 and fav_idx < len(pos_texts):
            fav_snippet = pos_texts[fav_idx].replace("\n", " ").strip()[:60] + "..."
            reason = f'Matches: "{fav_snippet}"'

        score_color = "green" if score > 0.6 else "yellow" if score > 0.4 else "white"
        score_str = f"[{score_color}]{score:.2f}[/{score_color}]"

        console.print(
            f"{score_str} [dim]({story.get('score', 0):4d})[/dim] [bold]{story.get('title', 'Untitled')}[/bold]"
        )

        if story.get("url"):
            console.print(f"   [dim cyan]Article:[/] {story['url']}")
        console.print(
            f"   [dim blue]Discuss:[/] https://news.ycombinator.com/item?id={story['id']}"
        )
        if reason:
            console.print(f"   [dim italic]{reason}[/]")

        console.print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HN Personalized Reranker")
    parser.add_argument(
        "username", nargs="?", help="Hacker News username (optional if saved)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of candidate stories to fetch (default: 100)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Time interval in days to search (default: 30)",
    )
    parser.add_argument(
        "--diversity", type=float, default=0.0, help="Diversity penalty (0.0-1.0)"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="Number of clusters for favorites (0 = off)",
    )
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=3,
        help="Max items to show per cluster (prevents topic flooding)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="onnx_model",
        help="Embedding model to use (default: onnx_model -> nomic-1.5)",
    )
    parser.add_argument(
        "--login", action="store_true", help="Login to Hacker News (for voting)"
    )
    parser.add_argument(
        "--tui", action="store_true", help="Launch interactive TUI mode"
    )

    args = parser.parse_args()

    if args.login:
        from api.client import HNClient
        import getpass

        u = input("Username: ")
        p = getpass.getpass("Password: ")

        async def run_login():
            client = HNClient()
            success, msg = await client.login(u, p)
            await client.close()
            if success:
                print(f"Success: {msg}")
                # Also save username to config
                save_config("username", u)
            else:
                print(f"Error: {msg}")
                exit(1)

        asyncio.run(run_login())
        exit(0)

    if args.tui:
        from tui_app import HNRerankTUI

        # Pass saved username if not provided
        u = args.username or get_saved_username()
        if not u:
            # Let TUI handle missing username (show input)
            u = ""

        app = HNRerankTUI(
            u, args.limit, args.days, args.model, diversity=args.diversity
        )
        app.run()
    else:
        asyncio.run(main(args))

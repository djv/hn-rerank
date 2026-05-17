#!/usr/bin/env -S uv run
import os
import praw
import json
from pathlib import Path

def fetch_reddit_signals():
    # Credentials from environment or .env
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    username = os.environ.get("REDDIT_USERNAME")
    password = os.environ.get("REDDIT_PASSWORD")

    if not all([client_id, client_secret, username, password]):
        print("Error: Missing REDDIT_ environment variables.")
        print("Required: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD")
        return

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        password=password,
        user_agent="hn-rerank-script:v1.0 (by /u/" + username + ")",
        username=username,
    )

    print(f"Fetching signals for /u/{username}...")
    
    signals = {
        "upvoted": [],
        "downvoted": []
    }

    # Fetch upvoted
    print("- Fetching upvoted submissions...")
    for submission in reddit.user.me().upvoted(limit=100):
        signals["upvoted"].append({
            "id": submission.id,
            "title": submission.title,
            "url": submission.url,
            "subreddit": submission.subreddit.display_name,
            "selftext": submission.selftext[:500] if submission.is_self else ""
        })

    # Fetch downvoted
    print("- Fetching downvoted submissions...")
    for submission in reddit.user.me().downvoted(limit=100):
        signals["downvoted"].append({
            "id": submission.id,
            "title": submission.title,
            "url": submission.url,
            "subreddit": submission.subreddit.display_name,
            "selftext": submission.selftext[:500] if submission.is_self else ""
        })

    output_path = Path(".cache/reddit_signals.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(signals, f, indent=2)
    
    print(f"\nSuccess! Saved {len(signals['upvoted'])} upvoted and {len(signals['downvoted'])} downvoted items to {output_path}")

if __name__ == "__main__":
    fetch_reddit_signals()

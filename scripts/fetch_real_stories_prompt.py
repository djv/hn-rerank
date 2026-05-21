#!/usr/bin/env -S uv run
import asyncio

from api.fetching import get_best_stories
from api.config import AppConfig
from api.llm_utils import _build_tldr_prompt

async def main():
    print("Fetching 3 real candidate stories...")
    # Get 3 candidate stories
    stories = await get_best_stories(limit=3, config=AppConfig(no_rss=True))
    
    if not stories:
        print("No stories found.")
        return

    stories_formatted = []
    for s in stories:
        sid = s.id
        title = s.title or "Untitled"
        comments = list(s.comments)
        text_content = s.text_content or ""
        
        context = f"### STORY ID: {sid} ###\nTitle: {title}"
        if text_content:
            context += f"\nContent: {text_content[:500]}"
        if comments:
            context += "\nComments:\n" + "\n".join(f"- {c[:200]}" for c in comments[:3])
        stories_formatted.append(context)

    prompt = _build_tldr_prompt(stories_formatted)
    
    with open("real_stories_prompt.txt", "w") as f:
        f.write(prompt)
    
    print(f"Generated prompt with {len(stories)} stories. Saved to real_stories_prompt.txt")

if __name__ == "__main__":
    asyncio.run(main())

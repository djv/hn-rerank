import asyncio
from api.main import get_user_data, get_best_stories
import json

async def debug():
    username = "pg"
    print(f"Fetching data for {username}...")
    try:
        pos, neg, exclude = await get_user_data(username)
        print(f"User Data: {len(pos)} positive, {len(neg)} negative, {len(exclude)} excluded")
        
        print("Fetching candidates...")
        candidates = await get_best_stories(100, 30, exclude)
        print(f"Found {len(candidates)} candidates")
        
        if candidates:
            print("First candidate:", candidates[0]['title'])
        else:
            print("No candidates found! Check Algolia connectivity or filters.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug())

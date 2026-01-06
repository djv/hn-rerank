import asyncio
import httpx
from datetime import datetime, timezone, timedelta


async def debug_algolia():
    days = 999
    limit = 1000
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    params = {
        "query": "",
        "tags": "story",
        "numericFilters": f"created_at_i>{start_time},points>20",
        "hitsPerPage": limit,
    }

    url = "https://hn.algolia.com/api/v1/search"
    print(f"Requesting {url} with params: {params}")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, params=params)
            print(f"Status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error Body: {resp.text}")
            else:
                data = resp.json()
                print(f"Hits: {len(data.get('hits', []))}")
        except Exception as e:
            print(f"Exception: {e}")


if __name__ == "__main__":
    asyncio.run(debug_algolia())

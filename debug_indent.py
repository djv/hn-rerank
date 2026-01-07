import httpx
from bs4 import BeautifulSoup
import asyncio

async def main():
    # Use a known story with nested comments
    story_id = 16582136 
    url = f"https://news.ycombinator.com/item?id={story_id}"
    print(f"Fetching {url}...")
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        
        rows = soup.find_all("tr", class_="comtr")
        print(f"Total rows: {len(rows)}")
        
        for i, row in enumerate(rows[:20]):
            ind_td = row.find("td", class_="ind")
            indent = int(ind_td.get("indent")) if ind_td else -1
            
            commtext = row.find(class_="commtext")
            text = commtext.get_text(strip=True)[:40] if commtext else "[No text]"
            
            print(f"#{i:02d} Indent: {indent} | {text}")

if __name__ == "__main__":
    asyncio.run(main())

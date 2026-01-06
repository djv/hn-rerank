
import asyncio
import time
import cProfile
import pstats
import io
from api.fetching import get_user_data, get_best_stories
from api import rerank
import os

async def profile_run():
    username = "pg"
    print(f"Profiling ranking for user: {username}")
    
    start_time = time.time()
    
    print("1. Fetching user data...")
    t0 = time.time()
    pos_data, neg_data, exclude_ids = await get_user_data(username)
    t1 = time.time()
    print(f"   User data fetched in {t1 - t0:.2f}s. Pos: {len(pos_data)}, Neg: {len(neg_data)}")
    
    print("2. Fetching candidates...")
    # Use a smaller count for profiling to be quicker, but large enough to measure
    # Using 200 instead of 1000 to save network time during dev profiling, 
    # but scaling up math helps. Let's use the constant from code if possible or just 200.
    # The user complained about "too long", so maybe 1000 is the issue.
    # Let's use 200 for now to check CPU bottlenecks vs Network.
    candidates_count = 200 
    t0 = time.time()
    candidates = await get_best_stories(candidates_count, 30, exclude_ids)
    t1 = time.time()
    print(f"   Fetched {len(candidates)} candidates in {t1 - t0:.2f}s")
    
    if not candidates:
        print("No candidates found.")
        return

    print("3. Ranking...")
    t0 = time.time()
    
    # We profile the ranking part specifically
    pr = cProfile.Profile()
    pr.enable()
    
    p_texts = [s.get("text_content", "") for s in pos_data]
    n_texts = [s.get("text_content", "") for s in neg_data]
    c_texts = [s.get("text_content", "") for s in candidates]

    p_timestamps = [s.get("time", 0) for s in pos_data]
    p_weights = (
        rerank.compute_recency_weights(p_timestamps)
        if p_timestamps
        else None
    )

    p_emb = rerank.get_embeddings(p_texts, is_query=True)
    n_emb = rerank.get_embeddings(n_texts, is_query=True)
    c_emb = rerank.get_embeddings(c_texts, is_query=False)

    ranked = rerank.rank_stories(
        candidates,
        cand_embeddings=c_emb,
        positive_embeddings=p_emb,
        negative_embeddings=n_emb,
        positive_weights=p_weights,
    )
    
    pr.disable()
    t1 = time.time()
    print(f"   Ranking completed in {t1 - t0:.2f}s")
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")

if __name__ == "__main__":
    # Ensure model is initialized
    rerank.init_model("onnx_model")
    asyncio.run(profile_run())

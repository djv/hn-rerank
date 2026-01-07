
from api.rerank import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def check():
    pairs = [
        (
            "Beginning January 2026, all ACM publications will be made open access",
            "2025: The Year in LLMs"
        ),
        (
            "Tell HN: Merry Christmas", 
            "The Most Popular Blogs of Hacker News in 2025"
        ),
        (
            "Inside CECOT – 60 Minutes",
            "Anna's Archive loses .org domain after surprise suspension"
        )
    ]
    
    print("Checking similarity scores...")
    for cand, fav in pairs:
        # Candidate (Title-only clean)
        v_cand = get_embeddings([cand], is_query=False)
        # Favorite (Title-only clean)
        v_fav = get_embeddings([fav], is_query=True)
        
        sim = cosine_similarity(v_fav, v_cand)[0][0]
        print(f"[{sim:.4f}] '{cand}' vs '{fav}'")

if __name__ == "__main__":
    check()

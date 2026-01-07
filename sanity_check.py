import numpy as np
from api.rerank import rank_stories

def test_ranking():
    # 2 Favorites: [1, 0], [0, 1]
    pos_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    
    # 3 Candidates:
    # 1. Exact match to F1
    # 2. 50/50 match
    # 3. No match (negative)
    stories = [
        {"id": 1, "score": 100, "time": 0, "text_content": "Match F1"},
        {"id": 2, "score": 100, "time": 0, "text_content": "Match both"},
        {"id": 3, "score": 100, "time": 0, "text_content": "No match"},
    ]
    
    cand_emb = np.array([
        [1.0, 0.0],
        [0.7, 0.7],
        [-1.0, 0.0]
    ], dtype=np.float32)
    
    # Mock model and get_embeddings
    import api.rerank
    from unittest.mock import MagicMock
    api.rerank.ONNXEmbeddingModel = MagicMock()
    api.rerank.get_embeddings = lambda texts, **kwargs: cand_emb
    
    results = rank_stories(stories, pos_emb, hn_weight=0.0, diversity_lambda=0.0)
    
    print("Ranked results:")
    for idx, score, fav_idx in results:
        print(f"Story {stories[idx]['id']}: score={score:.2f}, matched_fav={fav_idx}")
    
    assert results[0][0] == 0 # Story 1 should be first
    assert results[0][2] == 0 # Should match F1
    assert results[1][0] == 1 # Story 2 should be second
    print("Sanity check passed!")

if __name__ == "__main__":
    test_ranking()

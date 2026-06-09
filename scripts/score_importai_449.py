# ruff: noqa
"""Score Import AI 449 using the current model."""

from __future__ import annotations
import sys
import time

sys.path.insert(0, "/home/dev/hn_rerank")

from api.config import AppConfig
from api.feedback import load_feedback
from api.feedback_single_model import (
    build_single_model_feedback_labels,
    train_single_model_from_embeddings,
)
from api.models import Story
from api.rerank import get_embeddings, rank_stories

# Import AI 449 article text (from RSS feed paged=2)
ARTICLE_TEXT = """PostTrainBench shows startling growth in AI capabilities at post-training.
AI-driven R&D might be the most important thing in all of AI, because it helps us understand whether AI systems might eventually build their own successors. So far, much of the focus on AI R&D has been in components that support AI development (e.g., autonomous creation of AI kernels), or training base models. But there's been less attention paid to fine-tuning.
Researchers from the University of Tübingen, the Max Planck Institute for Intelligent Systems, and AI research organization Thoughtful Lab want to change that with PostTrainBench, a benchmark which targets a specific aspect of post-training; improving performance against a given dataset.
COVENANT-72B: Challenging the political economy of AI via distributed training. A bunch of people have used the blockchain to coordinate the distributed training run of a 72B parameter model which matches the performance of LLaMA2, a model trained and released by Facebook in 2023.
If AI writes all the world's software, we should invest more in verification. Leonardo de Moura thinks that the rise of AI for the creation of new software means that humans need to invest a lot more in verification.
Computer vision is a lot harder and less general than generative text. Facebook, the World Resources Institute, and the University of Maryland, have built CHMv2, a global, meter-resolution canopy height map.
Tech Tales: Singleton — 18 years after the pathological narcissus bomb which doomed the uplift."""


def main():
    config = AppConfig.load()

    print("Loading feedback...")
    feedback_records = load_feedback()
    print(f"  {len(feedback_records)} feedback records loaded")

    labels_result = build_single_model_feedback_labels(feedback_records)
    labels = labels_result.labels
    print(f"  {len(labels)} usable labels ({labels_result.skipped_count} skipped)")

    pos_stories = [item.story for item in labels if item.label == 2]
    neg_stories = [item.story for item in labels if item.label == 0]
    print(f"  {len(pos_stories)} positive, {len(neg_stories)} negative")

    print("Embedding feedback stories...")
    feedback_emb = get_embeddings([s.story.text_content for s in labels_result.labels])

    print("Embedding positive/negative references...")
    p_emb = (
        get_embeddings([s.text_content for s in pos_stories], is_query=True)
        if pos_stories
        else None
    )
    n_emb = (
        get_embeddings([s.text_content for s in neg_stories], is_query=True)
        if neg_stories
        else None
    )

    print("Training model...")
    model, _ = train_single_model_from_embeddings(
        labels, feedback_emb, p_emb, n_emb, config, config.single_model
    )

    # Create Import AI 449 story
    importai = Story(
        id=0,  # external story, no HN id
        title="ImportAI 449: LLMs training other LLMs; 72B distributed training run; computer vision is harder than generative text",
        url="https://jack-clark.net/2026/03/16/importai-449-llms-training-other-llms-72b-distributed-training-run-computer-vision-is-harder-than-generative-text/",
        score=0,
        time=int(time.time()),
        text_content=ARTICLE_TEXT,
        source="rss",
        comment_count=None,
    )

    print("Scoring Import AI 449...")
    results = rank_stories(
        [importai],
        model,
        p_emb,
        n_emb,
        config=config,
        positive_stories=pos_stories,
        negative_stories=neg_stories,
    )

    r = results[0]
    print("\nResults for Import AI 449:")
    print(f"  model_score:     {r.model_score:.4f}")
    print(f"  max_sim_score:   {r.max_sim_score:.4f}")
    print(f"  knn_score:       {r.knn_score:.4f}")
    print(f"  max_cluster_score: {r.max_cluster_score:.4f}")
    print(f"  p_up:            {r.p_up:.4f}")
    print(f"  p_neutral:       {r.p_neutral:.4f}")
    print(f"  p_down:          {r.p_down:.4f}")
    print(f"  entropy:         {r.entropy:.4f}")
    print(f"  passes 0.52 threshold? {'YES' if r.model_score >= 0.52 else 'NO'}")


if __name__ == "__main__":
    main()

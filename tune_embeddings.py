#!/usr/bin/env -S uv run
"""Fine-tune embeddings using user's positive/negative signals with contrastive learning."""

import argparse
import json
from pathlib import Path
import logging
from typing import cast

from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_triplets(path: Path) -> list[InputExample]:
    """Load triplet data (anchor, positive, negative)."""
    examples = []
    if not path.exists():
        return examples

    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        item = json.loads(line)
        examples.append(
            InputExample(
                texts=[item["anchor"], item["positive"], item["negative"]]
            )
        )
    return examples


def load_pairs(path: Path) -> list[InputExample]:
    """Load positive pairs (anchor, positive)."""
    examples = []
    if not path.exists():
        return examples

    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        item = json.loads(line)
        examples.append(InputExample(texts=[item["anchor"], item["positive"]]))
    return examples


def tune(epochs: int = 3, batch_size: int = 16, use_triplets: bool = True) -> None:
    """Fine-tune the embedding model."""

    # Check for data
    train_triplets_path = Path("train_triplets.jsonl")
    train_pairs_path = Path("train_pairs.jsonl")
    val_triplets_path = Path("val_triplets.jsonl")

    has_triplets = train_triplets_path.exists() and train_triplets_path.stat().st_size > 0
    has_pairs = train_pairs_path.exists() and train_pairs_path.stat().st_size > 0

    if not has_triplets and not has_pairs:
        # Fallback to legacy format
        legacy_train = Path("train.jsonl")
        if not legacy_train.exists():
            print("No training data found. Run: uv run prepare_data.py <username>")
            return
        print("Using legacy pair format...")
        has_pairs = True
        train_pairs_path = legacy_train

    # Load data
    train_examples = []

    if use_triplets and has_triplets:
        triplets = load_triplets(train_triplets_path)
        print(f"Loaded {len(triplets)} triplets for training")
        train_examples.extend(triplets)
    elif has_pairs:
        pairs = load_pairs(train_pairs_path)
        print(f"Loaded {len(pairs)} pairs for training")
        train_examples.extend(pairs)

    if not train_examples:
        print("No training examples loaded.")
        return

    # Load validation triplets for evaluation
    val_triplets = load_triplets(val_triplets_path) if val_triplets_path.exists() else []

    # Load model
    model_id = "BAAI/bge-base-en-v1.5"
    print(f"Loading {model_id}...")
    model = SentenceTransformer(model_id)

    # Setup training
    train_dataloader = DataLoader(
        cast(Dataset[InputExample], train_examples),
        shuffle=True,
        batch_size=batch_size,
    )

    # Choose loss based on data format
    if use_triplets and has_triplets:
        # TripletLoss with hard negatives - better for personalization
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.5,  # Margin between pos and neg
        )
        print("Using TripletLoss with hard negatives")
    else:
        # MultipleNegativesRankingLoss - uses in-batch negatives
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print("Using MultipleNegativesRankingLoss")

    # Setup evaluator
    evaluator = None
    if val_triplets:
        anchors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        for t in val_triplets:
            if not t.texts or len(t.texts) < 3:
                continue
            anchors.append(t.texts[0])
            positives.append(t.texts[1])
            negatives.append(t.texts[2])
        if anchors:
            evaluator = TripletEvaluator(
                anchors,
                positives,
                negatives,
                name="hn_triplet_eval",
                show_progress_bar=False,
            )

    # Training config
    warmup_steps = int(len(train_dataloader) * 0.1)
    eval_steps = max(1, len(train_dataloader) // 2)  # Evaluate twice per epoch

    print(f"Training for {epochs} epochs, batch size {batch_size}")
    print(f"Total steps: {len(train_dataloader) * epochs}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=eval_steps if evaluator else 0,
        warmup_steps=warmup_steps,
        output_path="tuned_model",
        save_best_model=bool(evaluator),
        show_progress_bar=True,
    )

    print("\nTraining complete!")
    print("Model saved to tuned_model/")
    print("Run 'uv run export_tuned.py' to convert to ONNX.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune embeddings")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--no-triplets", action="store_true", help="Use pairs only (no hard negatives)")
    args = parser.parse_args()

    tune(epochs=args.epochs, batch_size=args.batch_size, use_triplets=not args.no_triplets)

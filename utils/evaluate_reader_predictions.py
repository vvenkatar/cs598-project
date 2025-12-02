#!/usr/bin/env python3
"""
Evaluate reader predictions against gold answers.
Computes macro F1 and Exact Match (EM).
"""

import argparse
import json
import os
from statistics import mean
from typing import List, Tuple

from f1_score import calculate_f1


def load_predictions(path: str) -> List[dict]:
    """Load a JSON file and return the list of prediction objects."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prediction file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON structure in file: {path}")


def evaluate_entries(entries: List[dict]) -> Tuple[List[float], List[int]]:
    """Compute F1 and EM lists from prediction objects."""
    f1_scores: List[float] = []
    exact_matches: List[int] = []

    for entry in entries:
        try:
            gold_answer = entry["gold_answers"][0]
            pred_answer = entry["predictions"][0]["prediction"]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Entry missing required keys: {entry}")

        f1 = calculate_f1(gold_answer, pred_answer)
        f1_scores.append(f1)
        exact_matches.append(1 if f1 == 1.0 else 0)

    return f1_scores, exact_matches


def summarize_results(file_path: str) -> None:
    """Load, evaluate, and print summary metrics for a predictions file."""
    entries = load_predictions(file_path)
    f1_scores, exact_matches = evaluate_entries(entries)

    base = os.path.basename(file_path).split("_")[0]
    em = mean(exact_matches)
    f1 = mean(f1_scores)

    print(f"{base} → EM: {em:.2f}, F1: {f1:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reader predictions.")
    parser.add_argument("pred_file", type=str, help="Path to JSON prediction file.")
    return parser.parse_args()


def main():
    args = parse_args()
    summarize_results(args.pred_file)


if __name__ == "__main__":
    main()

"""Small, dependency-free classification metrics for the security model."""

from __future__ import annotations

from collections import defaultdict


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    """Integrate the precision-recall staircase, grouping equal scores."""
    positives = sum(labels)
    if not positives:
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    true_positive = false_positive = 0
    area = 0.0
    position = 0
    while position < len(ranked):
        score = ranked[position][0]
        group = []
        while position < len(ranked) and ranked[position][0] == score:
            group.append(ranked[position][1])
            position += 1
        old_recall = true_positive / positives
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        area += (true_positive / positives - old_recall) * true_positive / (true_positive + false_positive)
    return area


def evaluate_scores(scores: list[float], labels: list[int], threshold: float) -> dict:
    """Summarize binary decisions at one frozen policy threshold."""
    true_positive = sum(score >= threshold and label == 1 for score, label in zip(scores, labels))
    false_positive = sum(score >= threshold and label == 0 for score, label in zip(scores, labels))
    false_negative = sum(labels) - true_positive
    true_negative = len(labels) - sum(labels) - false_positive
    return {"support": len(labels), "positives": sum(labels),
            "confusion": {"tp": true_positive, "fp": false_positive,
                          "tn": true_negative, "fn": false_negative},
            "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
            "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
            "fpr": false_positive / (false_positive + true_negative) if false_positive + true_negative else None,
            "pr_auc": average_precision(scores, labels)}


def report_by_source(scores: list[float], labels: list[int], sources: list[str], threshold: float) -> dict:
    """Keep per-site reports separate from the global weighted average."""
    groups = defaultdict(lambda: ([], []))
    for score, label, source in zip(scores, labels, sources):
        groups[source][0].append(score)
        groups[source][1].append(label)
    return {source: evaluate_scores(*values, threshold) for source, values in groups.items()}

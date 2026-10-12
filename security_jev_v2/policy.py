"""Fit held-out score calibration and evaluate an experimental low-FPR policy."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

# These local metrics group tied scores and never select a threshold from test labels.
from security_jev_v2.metrics import evaluate_scores, report_by_source
from security_jev_v2.model import TYPE_NAMES


def _fit_binary(logits: list[float], labels: list[int]) -> tuple[float, float]:
    if len(logits) < 4 or len(set(labels)) < 2:
        return 1.0, 0.0
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / log_temperature.exp() + bias, y)
        (loss + .01 * (log_temperature.square() + bias.square())).backward()
        return loss + .01 * (log_temperature.square() + bias.square())

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(.2, 5)), float(bias.detach().clamp(-5, 5))


def _fit_choice(logits: list[list[float]], labels: list[int]) -> float:
    if len(logits) < 4 or len(set(labels)) < 2:
        return 1.0
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x / log_temperature.exp(), y) + .01 * log_temperature.square()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(.2, 5))


def calibrate(block_logits: list[float], block_labels: list[int],
              type_logits: list[list[float]], type_labels: list[int]) -> dict:
    """Only the calibration split determines post-hoc scaling parameters."""
    if len(block_logits) != len(block_labels) or len(type_logits) != len(type_labels):
        raise ValueError("calibration score/label lengths differ")
    block_temperature, block_bias = _fit_binary(block_logits, block_labels)
    return {"block_temperature": block_temperature, "block_bias": block_bias,
            "type_temperature": _fit_choice(type_logits, type_labels)}


def probabilities(block_logits: list[float], type_logits: list[list[float]], calibration: dict) -> tuple[list[float], list[list[float]]]:
    """Return calibrated per-question distributions without coupling the two heads."""
    blocks = torch.sigmoid(torch.tensor(block_logits, dtype=torch.float64) /
                           calibration["block_temperature"] + calibration["block_bias"]).tolist()
    types = (torch.softmax(torch.tensor(type_logits, dtype=torch.float64) /
                           calibration["type_temperature"], dim=-1).tolist() if type_logits else [])
    return blocks, types


def choose_policy(block_scores: list[float], block_labels: list[int], max_fpr: float = .01) -> dict:
    """Freeze a candidate action threshold using calibration labels only."""
    if len(block_scores) != len(block_labels) or not block_scores or not 0 <= max_fpr < 1:
        raise ValueError("invalid calibration scores")
    negatives = len(block_labels) - sum(block_labels)
    if negatives == 0:
        raise ValueError("policy selection needs benign calibration requests")
    # Sort once and move whole tied-score groups; this scales to large sites.
    budget = math.floor(max_fpr * negatives)
    threshold, best_tp = math.nextafter(max(block_scores), math.inf), 0
    fp = tp = 0
    ranked = sorted(zip(block_scores, block_labels), reverse=True)
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == score:
            tp += ranked[index][1]
            fp += 1 - ranked[index][1]
            index += 1
        if fp > budget:
            break
        if tp > best_tp:
            threshold, best_tp = score, tp
    result = evaluate_scores(block_scores, block_labels, threshold)
    return {"threshold": threshold, "max_fpr": max_fpr, "fpr": result["fpr"],
            "recall": result["recall"], "status": "offline_experiment_only"}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Represent sampling uncertainty; zero observed errors is not zero true risk."""
    if total <= 0:
        return None
    p = successes / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def _ece(confidence: list[float], correct: list[bool], bins: int = 10) -> float | None:
    if not confidence:
        return None
    error = 0.0
    for index in range(bins):
        selected = [i for i, p in enumerate(confidence)
                    if index / bins <= p < (index + 1) / bins or (index == bins - 1 and p == 1)]
        if selected:
            error += len(selected) / len(confidence) * abs(
                sum(confidence[i] for i in selected) / len(selected)
                - sum(correct[i] for i in selected) / len(selected))
    return error


def report_predictions(block_scores: list[float], block_labels: list[int],
                       type_probs: list[list[float]], type_labels: list[int],
                       sites: list[str], threshold: float) -> dict:
    """Report detection, conditional classification and end-to-end results."""
    if len(block_scores) != len(block_labels) or len(sites) != len(block_labels):
        raise ValueError("block report input lengths differ")
    attack_indices = [i for i, label in enumerate(block_labels) if label == 1]
    if len(type_probs) != len(type_labels) or len(type_probs) != len(attack_indices):
        raise ValueError("each evaluated attack needs one known Choice label")
    block = evaluate_scores(block_scores, block_labels, threshold)
    negative = len(block_labels) - sum(block_labels)
    block["fpr_wilson_95"] = wilson_interval(block["confusion"]["fp"], negative)
    block["brier"] = sum((p - y) ** 2 for p, y in zip(block_scores, block_labels)) / len(block_labels)
    block["ece_10"] = _ece([max(p, 1 - p) for p in block_scores],
                           [(p >= .5) == bool(y) for p, y in zip(block_scores, block_labels)])
    per_type = {}
    correct_count = 0
    end_to_end_count = 0
    for index, name in enumerate(TYPE_NAMES):
        rows = [j for j, label in enumerate(type_labels) if label == index]
        correct = sum(max(range(len(type_probs[j])), key=type_probs[j].__getitem__) == index for j in rows)
        joint = sum(block_scores[attack_indices[j]] >= threshold and
                    max(range(len(type_probs[j])), key=type_probs[j].__getitem__) == index for j in rows)
        per_type[name] = {"positive_support": len(rows), "choice_accuracy": correct / len(rows) if rows else None,
                          "end_to_end_recall": joint / len(rows) if rows else None,
                          "status": "evaluated" if len(rows) >= 50 else "unsupported"}
        correct_count += correct
        end_to_end_count += joint
    return {"block": block,
            "choice": {"known_support": len(type_labels), "top1_accuracy": correct_count / len(type_labels) if type_labels else None,
                       "end_to_end_recall": end_to_end_count / len(type_labels) if type_labels else None,
                       "ece_10": _ece([max(row) for row in type_probs],
                                      [max(range(len(row)), key=row.__getitem__) == y for row, y in zip(type_probs, type_labels)]),
                       "by_type": per_type},
            "by_site": report_by_source(block_scores, block_labels, sites, threshold)}

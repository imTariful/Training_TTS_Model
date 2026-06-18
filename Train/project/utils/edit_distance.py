from __future__ import annotations

from functools import lru_cache


def levenshtein_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current_row = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (source_char != target_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_distance(source: str, target: str) -> float:
    denominator = max(len(source), len(target), 1)
    return levenshtein_distance(source, target) / denominator


def levenshtein_tokens(source: list[str], target: list[str]) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_token in enumerate(source, start=1):
        current_row = [i]
        for j, target_token in enumerate(target, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (source_token != target_token)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def normalized_token_edit_distance(source: list[str], target: list[str]) -> float:
    denominator = max(len(source), len(target), 1)
    return levenshtein_tokens(source, target) / denominator

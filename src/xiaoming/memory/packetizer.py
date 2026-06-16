from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from xiaoming.memory.models import MemoryFragment


@dataclass(frozen=True)
class MemoryPacket:
    id: str
    fragments: list[MemoryFragment]

    @property
    def token_estimate(self) -> int:
        return sum(fragment.token_estimate for fragment in self.fragments)


def packetize_fragments(fragments: list[MemoryFragment], token_budget: int) -> list[MemoryPacket]:
    packets = _split(list(fragments), max(token_budget, 1))
    return [MemoryPacket(id=f"packet-{index + 1}", fragments=packet) for index, packet in enumerate(packets) if packet]


def _split(fragments: list[MemoryFragment], token_budget: int) -> list[list[MemoryFragment]]:
    if sum(fragment.token_estimate for fragment in fragments) <= token_budget or len(fragments) <= 1:
        return [fragments]
    split_index = _best_split_index(fragments)
    left = fragments[:split_index]
    right = fragments[split_index:]
    return _split(left, token_budget) + _split(right, token_budget)


def _best_split_index(fragments: list[MemoryFragment]) -> int:
    total_tokens = sum(fragment.token_estimate for fragment in fragments)
    best_index = 1
    best_score = float("-inf")
    left_tokens = 0
    for index in range(1, len(fragments)):
        left_tokens += fragments[index - 1].token_estimate
        right_tokens = total_tokens - left_tokens
        gap_seconds = _gap_seconds(fragments[index - 1], fragments[index])
        night_bonus = 3600 * 2 if _is_night_boundary(fragments[index - 1], fragments[index]) else 0
        imbalance_penalty = abs(left_tokens - right_tokens)
        score = gap_seconds + night_bonus - imbalance_penalty
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _gap_seconds(left: MemoryFragment, right: MemoryFragment) -> float:
    return max((_parse_datetime(right.created_at) - _parse_datetime(left.created_at)).total_seconds(), 0)


def _is_night_boundary(left: MemoryFragment, right: MemoryFragment) -> bool:
    left_dt = _parse_datetime(left.created_at)
    right_dt = _parse_datetime(right.created_at)
    return left_dt.hour >= 18 and 0 <= right_dt.hour <= 8


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)

from xiaoming.memory.models import MemoryFragment
from xiaoming.memory.packetizer import MemoryPacket, packetize_fragments


def fragment(fragment_id, created_at, tokens=10):
    return MemoryFragment(
        id=fragment_id,
        source_event_id=fragment_id,
        role_or_type="user_message",
        created_at=created_at,
        timezone="Asia/Shanghai",
        token_estimate=tokens,
        content=fragment_id,
    )


def test_packetizer_keeps_small_fragment_list_intact():
    fragments = [
        fragment("a", "2026-06-05T10:00:00+08:00", 10),
        fragment("b", "2026-06-05T10:05:00+08:00", 10),
    ]

    packets = packetize_fragments(fragments, token_budget=25)

    assert packets == [MemoryPacket(id="packet-1", fragments=fragments)]


def test_packetizer_splits_on_largest_time_gap():
    fragments = [
        fragment("a", "2026-06-05T10:00:00+08:00", 20),
        fragment("b", "2026-06-05T10:05:00+08:00", 20),
        fragment("c", "2026-06-05T18:00:00+08:00", 20),
    ]

    packets = packetize_fragments(fragments, token_budget=45)

    assert [[fragment.id for fragment in packet.fragments] for packet in packets] == [["a", "b"], ["c"]]


def test_packetizer_prefers_night_gap_over_similar_day_gap():
    fragments = [
        fragment("a", "2026-06-05T20:00:00+08:00", 20),
        fragment("b", "2026-06-06T02:00:00+08:00", 20),
        fragment("c", "2026-06-06T08:30:00+08:00", 20),
    ]

    packets = packetize_fragments(fragments, token_budget=45)

    assert [[fragment.id for fragment in packet.fragments] for packet in packets] == [["a"], ["b", "c"]]


def test_packetizer_avoids_extreme_imbalance_when_gaps_are_close():
    fragments = [
        fragment("a", "2026-06-05T10:00:00+08:00", 10),
        fragment("b", "2026-06-05T14:00:00+08:00", 10),
        fragment("c", "2026-06-05T18:30:00+08:00", 10),
        fragment("d", "2026-06-05T19:00:00+08:00", 10),
    ]

    packets = packetize_fragments(fragments, token_budget=25)

    assert [[fragment.id for fragment in packet.fragments] for packet in packets] == [["a", "b"], ["c", "d"]]

import hashlib

from eval_tool.eval_set import sample_records


def _circular(n_groups, n_rot=4):
    return [{"id": f"q{g}__r{k}", "metadata": {"circular_id": f"q{g}", "rotation": k}}
            for g in range(n_groups) for k in range(n_rot)]


def test_sampling_takes_rotation_groups_whole():
    picked = sample_records(_circular(20), 10, seed=42)
    groups = {}
    for r in picked:
        groups.setdefault(r["metadata"]["circular_id"], []).append(r["metadata"]["rotation"])
    assert all(sorted(v) == [0, 1, 2, 3] for v in groups.values())
    assert len(picked) == 12            # 10 asked, the third group is finished, not cut


def test_plain_records_sample_exactly_as_before():
    records = [{"id": f"r{i}"} for i in range(50)]
    old = sorted(records, key=lambda r: hashlib.sha256(f"42|{r['id']}".encode("utf-8")).digest())[:7]
    assert sample_records(records, 7, seed=42) == old

#!/usr/bin/env python3
"""Extract test vectors from Python hstrat for cross-validation with Rust.

Generates JSON fixture files under tests/fixtures/ that the Rust test suite
loads to verify compatibility with the Python reference implementation.

Policy retained ranks are deterministic and RNG-independent, so they are
the primary cross-validation target. MRCA vectors use columns with known
differentia. Serialization vectors record exact packet bytes.

NOTE: Python hstrat auto-deposits rank 0 on column creation, so
GetNumStrataDeposited() returns 1 before any explicit DepositStratum().
The Rust implementation starts at 0 and requires explicit deposit.
Vectors use `num_strata_deposited` as the canonical count.

NOTE: Python MRCA bounds return (lower_inclusive, upper_exclusive).
Rust returns (lower_inclusive, upper_inclusive). Tests must adjust.

Usage:
    python scripts/extract_test_vectors.py

Requires:
    pip install hstrat
"""

import json
import os
import sys
import traceback

from hstrat.genome_instrumentation import HereditaryStratigraphicColumn
from hstrat.stratum_retention_strategy import (
    fixed_resolution_algo,
    perfect_resolution_algo,
    nominal_resolution_algo,
    recency_proportional_resolution_algo,
    depth_proportional_resolution_algo,
    depth_proportional_resolution_tapered_algo,
    geom_seq_nth_root_algo,
    geom_seq_nth_root_tapered_algo,
    pseudostochastic_algo,
    recency_proportional_resolution_curbed_algo,
)
from hstrat.serialization import col_to_packet
from hstrat.phylogenetic_inference import (
    calc_rank_of_mrca_bounds_between,
    does_have_any_common_ancestor,
)


FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def col_strata(col):
    """Extract (ranks, differentiae) from a column in rank order."""
    ranks = sorted(col.IterRetainedRanks())
    diffs = [
        col.GetStratumAtColumnIndex(i).GetDifferentia()
        for i in range(col.GetNumStrataRetained())
    ]
    return ranks, diffs


# ─── Policy Vectors ─────────────────────────────────────────────────
def extract_policy_vectors():
    """For each policy, extract retained ranks at n=0..1000.

    Uses policy.IterRetainedRanks(n) directly — purely mathematical,
    independent of RNG. Must match Rust exactly.
    """
    out = os.path.join(FIXTURES, "policy_vectors")
    ensure_dir(out)

    policies = {
        "fixed_resolution_1": fixed_resolution_algo.Policy(1),
        "fixed_resolution_5": fixed_resolution_algo.Policy(5),
        "fixed_resolution_10": fixed_resolution_algo.Policy(10),
        "fixed_resolution_50": fixed_resolution_algo.Policy(50),
        "perfect_resolution": perfect_resolution_algo.Policy(),
        "nominal_resolution": nominal_resolution_algo.Policy(),
        "recency_proportional_1": recency_proportional_resolution_algo.Policy(1),
        "recency_proportional_3": recency_proportional_resolution_algo.Policy(3),
        "recency_proportional_10": recency_proportional_resolution_algo.Policy(10),
        "depth_proportional_1": depth_proportional_resolution_algo.Policy(1),
        "depth_proportional_5": depth_proportional_resolution_algo.Policy(5),
        "depth_proportional_10": depth_proportional_resolution_algo.Policy(10),
        "depth_proportional_tapered_1": depth_proportional_resolution_tapered_algo.Policy(1),
        "depth_proportional_tapered_5": depth_proportional_resolution_tapered_algo.Policy(5),
        "depth_proportional_tapered_10": depth_proportional_resolution_tapered_algo.Policy(10),
        "geometric_seq_nth_root_2_2": geom_seq_nth_root_algo.Policy(2, 2),
        "geometric_seq_nth_root_3_2": geom_seq_nth_root_algo.Policy(3, 2),
        "geometric_seq_nth_root_2_4": geom_seq_nth_root_algo.Policy(2, 4),
        "geometric_seq_nth_root_tapered_2_2": geom_seq_nth_root_tapered_algo.Policy(2, 2),
        "geometric_seq_nth_root_tapered_3_2": geom_seq_nth_root_tapered_algo.Policy(3, 2),
        "curbed_recency_proportional_10": recency_proportional_resolution_curbed_algo.Policy(10),
        "curbed_recency_proportional_67": recency_proportional_resolution_curbed_algo.Policy(67),
    }

    max_n = 1000

    for name, policy in policies.items():
        data = {
            "policy": name,
            "max_n": max_n,
            "retained_ranks": {},
            "num_strata_retained": {},
        }

        for n in range(max_n + 1):
            ranks = sorted(policy.IterRetainedRanks(n))
            count = policy.CalcNumStrataRetainedExact(n)
            data["retained_ranks"][str(n)] = ranks
            data["num_strata_retained"][str(n)] = count

            assert len(ranks) == count, (
                f"{name} n={n}: len(ranks)={len(ranks)} != count={count}"
            )

        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(data, f)

    print(f"  policy_vectors: {len(policies)} policies, n=0..{max_n}")


# ─── Column Vectors ─────────────────────────────────────────────────
def extract_column_vectors():
    """Column retained rank sets after N deposits.

    Since Rust and Python use different RNGs, we verify that the column
    retains the correct set of ranks after a given number of deposits.
    """
    out = os.path.join(FIXTURES, "column_vectors")
    ensure_dir(out)

    test_cases = [
        ("perfect_n100", perfect_resolution_algo.Policy(), "perfect_resolution", 100),
        ("fixed10_n100", fixed_resolution_algo.Policy(10), "fixed_resolution_10", 100),
        ("fixed10_n500", fixed_resolution_algo.Policy(10), "fixed_resolution_10", 500),
        ("nominal_n50", nominal_resolution_algo.Policy(), "nominal_resolution", 50),
        ("recency3_n200", recency_proportional_resolution_algo.Policy(3), "recency_proportional_3", 200),
        ("depth5_n200", depth_proportional_resolution_algo.Policy(5), "depth_proportional_5", 200),
        ("geom22_n200", geom_seq_nth_root_algo.Policy(2, 2), "geometric_seq_nth_root_2_2", 200),
    ]

    results = []
    for name, policy, policy_name, n in test_cases:
        col = HereditaryStratigraphicColumn(
            stratum_retention_policy=policy,
            stratum_differentia_bit_width=64,
        )
        # Python auto-deposits rank 0, so deposit n-1 more
        for _ in range(n - 1):
            col.DepositStratum()

        assert col.GetNumStrataDeposited() == n, (
            f"{name}: expected {n}, got {col.GetNumStrataDeposited()}"
        )

        ranks = sorted(col.IterRetainedRanks())
        num_retained = col.GetNumStrataRetained()

        results.append({
            "name": name,
            "policy": policy_name,
            "num_strata_deposited": n,
            "num_strata_retained": num_retained,
            "retained_ranks": ranks,
        })

    with open(os.path.join(out, "column_vectors.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  column_vectors: {len(results)} test cases")


# ─── MRCA Vectors ────────────────────────────────────────────────────
def extract_mrca_vectors():
    """MRCA bounds for known column pairs.

    Records each column's exact ranks and differentiae so the Rust test
    can reconstruct them via from_parts() and verify MRCA calculations.

    NOTE: Python returns (lower_inclusive, upper_exclusive).
    Rust returns (lower_inclusive, upper_inclusive).
    The fixture records Python's convention; Rust tests must convert.
    """
    out = os.path.join(FIXTURES, "mrca_vectors")
    ensure_dir(out)

    results = []

    # Case 1: parent-child, perfect resolution
    parent = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=64,
    )
    for _ in range(19):  # total deposited = 20
        parent.DepositStratum()
    child = parent.CloneDescendant()
    for _ in range(9):  # child total = 30
        child.DepositStratum()

    p_ranks, p_diffs = col_strata(parent)
    c_ranks, c_diffs = col_strata(child)
    bounds = calc_rank_of_mrca_bounds_between(parent, child, prior="arbitrary")

    results.append({
        "name": "parent_child_perfect",
        "a_ranks": p_ranks,
        "a_differentiae": p_diffs,
        "a_num_deposited": parent.GetNumStrataDeposited(),
        "b_ranks": c_ranks,
        "b_differentiae": c_diffs,
        "b_num_deposited": child.GetNumStrataDeposited(),
        "bit_width": 64,
        "mrca_lower_inclusive": bounds[0] if bounds else None,
        "mrca_upper_exclusive": bounds[1] if bounds else None,
        "has_common_ancestor": does_have_any_common_ancestor(parent, child),
    })

    # Case 2: siblings, fixed resolution 5
    ancestor = HereditaryStratigraphicColumn(
        stratum_retention_policy=fixed_resolution_algo.Policy(5),
        stratum_differentia_bit_width=64,
    )
    for _ in range(49):
        ancestor.DepositStratum()

    sib_a = ancestor.CloneDescendant()
    for _ in range(19):
        sib_a.DepositStratum()
    sib_b = ancestor.CloneDescendant()
    for _ in range(29):
        sib_b.DepositStratum()

    sa_ranks, sa_diffs = col_strata(sib_a)
    sb_ranks, sb_diffs = col_strata(sib_b)
    bounds_sib = calc_rank_of_mrca_bounds_between(sib_a, sib_b, prior="arbitrary")

    results.append({
        "name": "siblings_fixed5",
        "a_ranks": sa_ranks,
        "a_differentiae": sa_diffs,
        "a_num_deposited": sib_a.GetNumStrataDeposited(),
        "b_ranks": sb_ranks,
        "b_differentiae": sb_diffs,
        "b_num_deposited": sib_b.GetNumStrataDeposited(),
        "bit_width": 64,
        "mrca_lower_inclusive": bounds_sib[0] if bounds_sib else None,
        "mrca_upper_exclusive": bounds_sib[1] if bounds_sib else None,
        "has_common_ancestor": does_have_any_common_ancestor(sib_a, sib_b),
    })

    # Case 3: diverge right after clone
    col_a = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=64,
    )
    for _ in range(4):
        col_a.DepositStratum()
    col_b = col_a.CloneDescendant()
    for _ in range(4):
        col_a.DepositStratum()
    for _ in range(4):
        col_b.DepositStratum()

    ca_ranks, ca_diffs = col_strata(col_a)
    cb_ranks, cb_diffs = col_strata(col_b)
    bounds_div = calc_rank_of_mrca_bounds_between(col_a, col_b, prior="arbitrary")

    results.append({
        "name": "diverge_after_clone",
        "a_ranks": ca_ranks,
        "a_differentiae": ca_diffs,
        "a_num_deposited": col_a.GetNumStrataDeposited(),
        "b_ranks": cb_ranks,
        "b_differentiae": cb_diffs,
        "b_num_deposited": col_b.GetNumStrataDeposited(),
        "bit_width": 64,
        "mrca_lower_inclusive": bounds_div[0] if bounds_div else None,
        "mrca_upper_exclusive": bounds_div[1] if bounds_div else None,
        "has_common_ancestor": does_have_any_common_ancestor(col_a, col_b),
    })

    with open(os.path.join(out, "mrca_vectors.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  mrca_vectors: {len(results)} test cases")


# ─── Serialization Vectors ───────────────────────────────────────────
def extract_serialization_vectors():
    """Record exact packet bytes for columns with known state.

    The Rust test constructs columns with the same rank/differentia via
    from_parts() and verifies col_to_packet produces identical bytes.
    """
    out = os.path.join(FIXTURES, "serialization_vectors")
    ensure_dir(out)

    results = []

    # perfect resolution, 64-bit, 10 strata
    col = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=64,
    )
    for _ in range(9):
        col.DepositStratum()
    ranks, diffs = col_strata(col)
    packet = col_to_packet(col)
    results.append({
        "name": "perfect_64bit_n10",
        "policy": "perfect_resolution",
        "bit_width": 64,
        "num_strata_deposited": col.GetNumStrataDeposited(),
        "retained_ranks": ranks,
        "retained_differentiae": diffs,
        "packet_hex": packet.hex(),
        "packet_len": len(packet),
    })

    # fixed resolution 10, 8-bit, 100 strata
    col2 = HereditaryStratigraphicColumn(
        stratum_retention_policy=fixed_resolution_algo.Policy(10),
        stratum_differentia_bit_width=8,
    )
    for _ in range(99):
        col2.DepositStratum()
    ranks2, diffs2 = col_strata(col2)
    packet2 = col_to_packet(col2)
    results.append({
        "name": "fixed10_8bit_n100",
        "policy": "fixed_resolution_10",
        "bit_width": 8,
        "num_strata_deposited": col2.GetNumStrataDeposited(),
        "retained_ranks": ranks2,
        "retained_differentiae": diffs2,
        "packet_hex": packet2.hex(),
        "packet_len": len(packet2),
    })

    # perfect resolution, 1-bit, 20 strata
    col3 = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=1,
    )
    for _ in range(19):
        col3.DepositStratum()
    ranks3, diffs3 = col_strata(col3)
    packet3 = col_to_packet(col3)
    results.append({
        "name": "perfect_1bit_n20",
        "policy": "perfect_resolution",
        "bit_width": 1,
        "num_strata_deposited": col3.GetNumStrataDeposited(),
        "retained_ranks": ranks3,
        "retained_differentiae": diffs3,
        "packet_hex": packet3.hex(),
        "packet_len": len(packet3),
    })

    with open(os.path.join(out, "serialization_vectors.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  serialization_vectors: {len(results)} test cases")


# ─── Tree Vectors ────────────────────────────────────────────────────
def extract_tree_vectors():
    """Record population states for tree reconstruction tests.

    Records each organism's ranks and differentiae so Rust can reconstruct
    identical columns via from_parts() and build trees. We record the
    population state; tree topology is validated by internal consistency
    (both Rust algorithms produce same output, property-based invariants).

    NOTE: Python's build_tree has a numpy compat issue on Python 3.14,
    so we only record population states, not expected tree output.
    """
    out = os.path.join(FIXTURES, "tree_vectors")
    ensure_dir(out)

    results = []

    # Case 1: 5 siblings from same ancestor, perfect resolution
    ancestor = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=64,
    )
    for _ in range(4):
        ancestor.DepositStratum()

    pop = []
    for _ in range(5):
        child = ancestor.CloneDescendant()
        for _ in range(2):
            child.DepositStratum()
        pop.append(child)

    pop_data = []
    for col in pop:
        ranks, diffs = col_strata(col)
        pop_data.append({
            "ranks": ranks,
            "differentiae": diffs,
            "num_strata_deposited": col.GetNumStrataDeposited(),
        })

    results.append({
        "name": "5_siblings_perfect",
        "population": pop_data,
        "bit_width": 64,
        "num_organisms": 5,
    })

    # Case 2: two-cluster tree
    root = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=64,
    )
    for _ in range(2):
        root.DepositStratum()

    branch_a = root.CloneDescendant()
    for _ in range(2):
        branch_a.DepositStratum()
    branch_b = root.CloneDescendant()
    for _ in range(2):
        branch_b.DepositStratum()

    pop2 = []
    for _ in range(3):
        child = branch_a.CloneDescendant()
        child.DepositStratum()
        pop2.append(child)
    for _ in range(3):
        child = branch_b.CloneDescendant()
        child.DepositStratum()
        pop2.append(child)

    pop2_data = []
    for col in pop2:
        ranks, diffs = col_strata(col)
        pop2_data.append({
            "ranks": ranks,
            "differentiae": diffs,
            "num_strata_deposited": col.GetNumStrataDeposited(),
        })

    results.append({
        "name": "two_clusters",
        "population": pop2_data,
        "bit_width": 64,
        "num_organisms": 6,
    })

    # Case 3: organisms with different depths
    shallow_anc = HereditaryStratigraphicColumn(
        stratum_retention_policy=perfect_resolution_algo.Policy(),
        stratum_differentia_bit_width=64,
    )
    for _ in range(2):
        shallow_anc.DepositStratum()

    pop3 = []
    for extra in [0, 2, 5, 10]:
        child = shallow_anc.CloneDescendant()
        for _ in range(extra):
            child.DepositStratum()
        pop3.append(child)

    pop3_data = []
    for col in pop3:
        ranks, diffs = col_strata(col)
        pop3_data.append({
            "ranks": ranks,
            "differentiae": diffs,
            "num_strata_deposited": col.GetNumStrataDeposited(),
        })

    results.append({
        "name": "different_depths",
        "population": pop3_data,
        "bit_width": 64,
        "num_organisms": 4,
    })

    with open(os.path.join(out, "tree_vectors.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  tree_vectors: {len(results)} population test cases")


def main():
    print("Extracting test vectors from Python hstrat...")
    print()

    extract_policy_vectors()
    extract_column_vectors()
    extract_mrca_vectors()
    extract_serialization_vectors()
    extract_tree_vectors()

    print()
    print("Done! Fixtures written to tests/fixtures/")


if __name__ == "__main__":
    main()

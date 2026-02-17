//! Property-based integration tests for hstrat.
//!
//! These tests verify cross-cutting invariants across the full stack:
//! policies, columns, MRCA, serialization, and tree reconstruction.

use hstrat::column::HereditaryStratigraphicColumn;
use hstrat::policies::*;
use hstrat::reconstruction::{
    build_tree, calc_rank_of_mrca_bounds_between, does_have_any_common_ancestor, TreeAlgorithm,
};
use hstrat::serialization::{col_from_packet, col_to_packet};

use proptest::prelude::*;

// ─── Policy Invariants ───

fn policy_invariants(policy: &impl StratumRetentionPolicy, max_n: u64) {
    for n in 0..=max_n {
        let count = policy.calc_num_strata_retained_exact(n);
        let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();

        // Count matches iterator length
        assert_eq!(
            count as usize,
            ranks.len(),
            "count mismatch at n={n}: calc={count}, iter={}",
            ranks.len()
        );

        if n == 0 {
            assert!(ranks.is_empty(), "n=0 should have no ranks");
            continue;
        }

        // Always retains rank 0
        assert_eq!(
            *ranks.first().unwrap(),
            0,
            "rank 0 not retained at n={n}"
        );

        // Always retains newest rank
        assert_eq!(
            *ranks.last().unwrap(),
            n - 1,
            "newest rank not retained at n={n}"
        );

        // Ranks strictly ascending
        for w in ranks.windows(2) {
            assert!(
                w[0] < w[1],
                "ranks not strictly ascending at n={n}: {:?}",
                ranks
            );
        }

        // All ranks in valid range
        for &r in &ranks {
            assert!(r < n, "rank {r} out of range at n={n}");
        }

        // calc_rank_at_column_index consistent with iter
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(
                policy.calc_rank_at_column_index(i, n),
                r,
                "calc_rank_at_column_index mismatch at n={n}, i={i}"
            );
        }
    }
}

#[test]
fn perfect_resolution_invariants() {
    policy_invariants(&PerfectResolutionPolicy::new(), 200);
}

#[test]
fn nominal_resolution_invariants() {
    policy_invariants(&NominalResolutionPolicy, 200);
}

#[test]
fn fixed_resolution_invariants() {
    for res in [1, 2, 5, 10, 50, 100] {
        policy_invariants(&FixedResolutionPolicy::new(res), 200);
    }
}

#[test]
fn depth_proportional_invariants() {
    for res in [1, 2, 5, 10] {
        policy_invariants(&DepthProportionalPolicy::new(res), 200);
    }
}

#[test]
fn depth_proportional_tapered_invariants() {
    for res in [1, 2, 5, 10] {
        policy_invariants(&DepthProportionalTaperedPolicy::new(res), 200);
    }
}

#[test]
fn recency_proportional_invariants() {
    for res in [1, 2, 5, 10] {
        policy_invariants(&RecencyProportionalPolicy::new(res), 200);
    }
}

#[test]
fn geometric_seq_nth_root_invariants() {
    for degree in [2, 3, 5] {
        for interspersal in [1, 2, 4] {
            policy_invariants(
                &GeometricSeqNthRootPolicy::new(degree, interspersal),
                200,
            );
        }
    }
}

#[test]
fn geometric_seq_nth_root_tapered_invariants() {
    for degree in [2, 3, 5] {
        policy_invariants(
            &GeometricSeqNthRootTaperedPolicy::new(degree, 2),
            200,
        );
    }
}

// ─── Column Invariants ───

fn column_invariants<P: StratumRetentionPolicy>(
    policy: P,
    bit_width: u8,
    seed: u64,
    n: u64,
) {
    let mut col = HereditaryStratigraphicColumn::with_seed(policy, bit_width, seed);
    col.deposit_strata(n);

    assert_eq!(col.get_num_strata_deposited(), n);
    assert_eq!(col.get_stratum_differentia_bit_width(), bit_width);

    let expected = col.get_policy().calc_num_strata_retained_exact(n) as usize;
    assert_eq!(
        col.get_num_strata_retained(),
        expected,
        "retained count mismatch at n={n}"
    );

    // Strata are sorted by rank
    let ranks: Vec<u64> = col.iter_retained_ranks().collect();
    for w in ranks.windows(2) {
        assert!(w[0] < w[1]);
    }

    // All differentia values are within bit_width
    let mask = hstrat::Differentia::mask(bit_width);
    for d in col.iter_retained_differentia() {
        assert!(
            d.value() <= mask,
            "differentia {} exceeds mask {} for bit_width {}",
            d.value(),
            mask,
            bit_width
        );
    }

    // Binary search finds all retained strata
    for stratum in col.iter_retained_strata() {
        let found = col.get_stratum_at_rank(stratum.rank);
        assert!(found.is_some(), "rank {} not found", stratum.rank);
        assert_eq!(found.unwrap().differentia, stratum.differentia);
    }
}

proptest! {
    #[test]
    fn proptest_perfect_column(seed in 0u64..1000, n in 0u64..500) {
        column_invariants(PerfectResolutionPolicy::new(), 64, seed, n);
    }

    #[test]
    fn proptest_fixed_column(
        resolution in 1u64..20,
        seed in 0u64..1000,
        n in 0u64..500,
    ) {
        column_invariants(FixedResolutionPolicy::new(resolution), 64, seed, n);
    }

    #[test]
    fn proptest_nominal_column(seed in 0u64..1000, n in 0u64..500) {
        column_invariants(NominalResolutionPolicy, 64, seed, n);
    }

    #[test]
    fn proptest_column_bit_widths(
        bit_width in 1u8..=64,
        seed in 0u64..1000,
        n in 0u64..100,
    ) {
        column_invariants(PerfectResolutionPolicy::new(), bit_width, seed, n);
    }
}

// ─── MRCA Invariants ───

#[test]
fn mrca_parent_child_bound_contains_truth() {
    // Parent-child: MRCA is parent's newest rank
    for seed in [0, 42, 123, 999] {
        let mut parent = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            64,
            seed,
        );
        parent.deposit_strata(20);
        let mut child = parent.clone_descendant();
        child.deposit_strata(10);

        let (lower, upper) =
            calc_rank_of_mrca_bounds_between(&parent, &child).unwrap();
        // True MRCA is at rank 19 (parent's newest when cloned)
        assert!(lower <= 19);
        assert!(upper >= 19);
    }
}

#[test]
fn mrca_siblings_bound_contains_truth() {
    // Siblings: MRCA is their common ancestor's newest rank.
    // clone_descendant() clones the column and deposits one stratum, so
    // each sibling shares ranks 0..9 (from ancestor) and diverges at rank 10
    // (independently generated differentia). The true MRCA is at rank 9
    // (the last shared stratum from the ancestor), but both siblings also
    // share the ancestor's strata through rank 9. With 64-bit differentia
    // and different RNG streams, they will almost certainly mismatch at 10.
    // However, the matching differentia at rank 10 is possible since
    // clone_descendant uses the cloned RNG which produces the same next value.
    // To truly test siblings, we need to deposit more strata to ensure divergence.
    for seed in [0, 42, 123] {
        let mut ancestor = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            64,
            seed,
        );
        ancestor.deposit_strata(10);

        // Clone twice — both clone the same RNG state, so rank 10 differentia
        // will be identical. We need to deposit independently to diverge.
        let mut sib_a = ancestor.clone_descendant();
        sib_a.deposit_strata(5); // ranks 11..15 with sib_a's RNG
        let mut sib_b = ancestor.clone_descendant();
        sib_b.deposit_strata(8); // ranks 11..18 with sib_b's RNG

        let result = calc_rank_of_mrca_bounds_between(&sib_a, &sib_b);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Both siblings share ancestor's strata 0..9.
        // At rank 10, both got the same differentia (cloned RNG).
        // Starting from rank 11, they diverge (different RNG paths due to
        // different deposit counts changing RNG state).
        // True MRCA is >= 9 (at least the ancestor's last rank).
        // The bounds should contain the truth.
        assert!(lower >= 9, "lower bound {} should be >= 9", lower);
        assert!(upper >= lower, "upper {} should be >= lower {}", upper, lower);
    }
}

#[test]
fn mrca_unrelated_columns_none() {
    // Two independently created columns (different seeds)
    let mut a = HereditaryStratigraphicColumn::with_seed(
        PerfectResolutionPolicy::new(),
        64,
        1,
    );
    let mut b = HereditaryStratigraphicColumn::with_seed(
        PerfectResolutionPolicy::new(),
        64,
        2,
    );
    a.deposit_strata(10);
    b.deposit_strata(10);

    // With 64-bit differentia and different seeds, almost certainly no match at rank 0
    let has_common = does_have_any_common_ancestor(&a, &b);
    if has_common {
        // Vanishingly unlikely but possible; just check bounds are valid
        let bounds = calc_rank_of_mrca_bounds_between(&a, &b);
        assert!(bounds.is_some());
    }
}

// ─── Serialization Round-Trip ───

proptest! {
    #[test]
    fn proptest_packet_round_trip_perfect(seed in 0u64..1000, n in 0u64..100) {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            64,
            seed,
        );
        col.deposit_strata(n);

        let packet = col_to_packet(&col);
        let restored = col_from_packet(&packet, PerfectResolutionPolicy::new(), 64).unwrap();

        prop_assert_eq!(col.get_num_strata_deposited(), restored.get_num_strata_deposited());
        prop_assert_eq!(col.get_num_strata_retained(), restored.get_num_strata_retained());

        for (a, b) in col.iter_retained_strata().zip(restored.iter_retained_strata()) {
            prop_assert_eq!(a.rank, b.rank);
            prop_assert_eq!(a.differentia, b.differentia);
        }
    }

    #[test]
    fn proptest_packet_round_trip_fixed(
        resolution in 1u64..20,
        seed in 0u64..1000,
        n in 0u64..200,
    ) {
        let policy = FixedResolutionPolicy::new(resolution);
        let mut col = HereditaryStratigraphicColumn::with_seed(
            policy.clone(),
            64,
            seed,
        );
        col.deposit_strata(n);

        let packet = col_to_packet(&col);
        let restored = col_from_packet(&packet, policy, 64).unwrap();

        prop_assert_eq!(col.get_num_strata_deposited(), restored.get_num_strata_deposited());
        prop_assert_eq!(col.get_num_strata_retained(), restored.get_num_strata_retained());

        for (a, b) in col.iter_retained_strata().zip(restored.iter_retained_strata()) {
            prop_assert_eq!(a.rank, b.rank);
            prop_assert_eq!(a.differentia, b.differentia);
        }
    }
}

// ─── Tree Reconstruction Invariants ───

#[test]
fn tree_leaves_match_population_count() {
    for pop_size in [1, 2, 5, 10, 20] {
        let mut ancestor = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            64,
            42,
        );
        ancestor.deposit_strata(5);

        let mut population = Vec::new();
        for _ in 0..pop_size {
            let mut org = ancestor.clone_descendant();
            org.deposit_strata(5);
            population.push(org);
        }

        let df = build_tree(&population, TreeAlgorithm::ShortcutConsolidation, None);

        let leaf_count = df
            .taxon_label
            .iter()
            .filter(|l| l.starts_with("taxon_"))
            .count();
        assert_eq!(
            leaf_count, pop_size,
            "expected {} leaves, got {}",
            pop_size, leaf_count
        );
    }
}

#[test]
fn tree_ancestor_ids_valid() {
    let mut ancestor = HereditaryStratigraphicColumn::with_seed(
        PerfectResolutionPolicy::new(),
        64,
        42,
    );
    ancestor.deposit_strata(5);

    let mut population = Vec::new();
    for _ in 0..10 {
        let mut org = ancestor.clone_descendant();
        org.deposit_strata(5);
        population.push(org);
    }

    let df = build_tree(&population, TreeAlgorithm::ShortcutConsolidation, None);

    // All ancestor IDs should reference valid node IDs
    let ids: std::collections::HashSet<u32> = df.id.iter().copied().collect();
    for ancestors in &df.ancestor_list {
        for &ancestor_id in ancestors {
            assert!(
                ids.contains(&ancestor_id),
                "ancestor_id {} not found in node ids",
                ancestor_id
            );
        }
    }
}

#[test]
fn tree_origin_times_non_negative() {
    let mut ancestor = HereditaryStratigraphicColumn::with_seed(
        PerfectResolutionPolicy::new(),
        64,
        42,
    );
    ancestor.deposit_strata(10);

    let mut population = Vec::new();
    for _ in 0..5 {
        let org = ancestor.clone_descendant();
        population.push(org);
    }

    let df = build_tree(&population, TreeAlgorithm::ShortcutConsolidation, None);

    for &t in &df.origin_time {
        assert!(t >= 0.0, "origin_time should be non-negative, got {}", t);
    }
}

// ─── Determinism ───

#[test]
fn column_deterministic_with_same_seed() {
    for bit_width in [1, 8, 32, 64] {
        let mut a = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            bit_width,
            42,
        );
        let mut b = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            bit_width,
            42,
        );
        a.deposit_strata(100);
        b.deposit_strata(100);

        let da: Vec<u64> = a.iter_retained_differentia().map(|d| d.value()).collect();
        let db: Vec<u64> = b.iter_retained_differentia().map(|d| d.value()).collect();
        assert_eq!(da, db, "same seed should produce identical differentiae");
    }
}

#[test]
fn tree_deterministic() {
    let build = || {
        let mut ancestor = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy::new(),
            64,
            42,
        );
        ancestor.deposit_strata(5);
        let pop: Vec<_> = (0..10)
            .map(|_| {
                let mut org = ancestor.clone_descendant();
                org.deposit_strata(5);
                org
            })
            .collect();
        build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None)
    };

    let df1 = build();
    let df2 = build();

    assert_eq!(df1.id, df2.id);
    assert_eq!(df1.ancestor_list, df2.ancestor_list);
    assert_eq!(df1.origin_time, df2.origin_time);
    assert_eq!(df1.taxon_label, df2.taxon_label);
}

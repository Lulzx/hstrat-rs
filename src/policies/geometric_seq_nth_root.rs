use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Retains strata using a geometric sequence based on the nth-root of the
/// number of depositions.
///
/// The `degree` parameter controls the exponent, and `interspersal` controls
/// how many strata are retained between each geometric target point.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricSeqNthRootPolicy {
    pub degree: u64,
    pub interspersal: u64,
}

impl GeometricSeqNthRootPolicy {
    pub fn new(degree: u64, interspersal: u64) -> Self {
        assert!(degree > 0, "degree must be positive");
        assert!(interspersal > 0, "interspersal must be positive");
        Self {
            degree,
            interspersal,
        }
    }

    /// Create with default interspersal of 2.
    pub fn with_degree(degree: u64) -> Self {
        Self::new(degree, 2)
    }
}

/// Compute the set of retained ranks for this policy.
fn compute_retained_ranks(
    degree: u64,
    interspersal: u64,
    num_strata_deposited: u64,
) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }
    if num_strata_deposited == 1 {
        return alloc::vec![0];
    }

    let newest_rank = num_strata_deposited - 1;
    let n = num_strata_deposited as f64;

    // Always retain the first `degree` strata and the newest rank.
    let mut retained = alloc::collections::BTreeSet::new();
    for r in 0..degree.min(num_strata_deposited) {
        retained.insert(r);
    }
    retained.insert(newest_rank);

    if num_strata_deposited <= degree + 1 {
        return retained.into_iter().collect();
    }

    // common_ratio = (n-1)^(1/degree)
    let common_ratio = libm::pow(n - 1.0, 1.0 / degree as f64);

    // Compute target recencies and corresponding target ranks.
    // target_recencies[i] = common_ratio^(i+1) for i in 0..degree
    let mut target_ranks = Vec::with_capacity(degree as usize + 1);
    target_ranks.push(newest_rank); // recency 1 => newest_rank itself

    for pow in 1..=degree {
        let recency = libm::pow(common_ratio, pow as f64);
        let recency_ceil = libm::ceil(recency) as u64;
        let target = if recency_ceil > newest_rank {
            0
        } else {
            newest_rank - recency_ceil
        };
        target_ranks.push(target);
    }

    // Sort target ranks ascending for interval computation.
    target_ranks.sort_unstable();
    target_ranks.dedup();

    // For each pair of consecutive target ranks, intersperse retained strata.
    for window in target_ranks.windows(2) {
        let lo = window[0];
        let hi = window[1];
        if hi <= lo {
            continue;
        }
        let span = hi - lo;
        if interspersal == 0 || span == 0 {
            retained.insert(lo);
            retained.insert(hi);
            continue;
        }
        // Place `interspersal` evenly-spaced strata in [lo, hi].
        // Always include lo and hi, plus intermediate points.
        let num_points = interspersal.min(span + 1);
        if num_points <= 1 {
            retained.insert(lo);
            retained.insert(hi);
        } else {
            for i in 0..num_points {
                let rank = lo + (i * span) / (num_points - 1);
                retained.insert(rank);
            }
        }
    }

    // Also retain the endpoints of the target list.
    for &tr in &target_ranks {
        retained.insert(tr);
    }

    retained.into_iter().collect()
}

impl StratumRetentionPolicy for GeometricSeqNthRootPolicy {
    fn gen_drop_ranks(
        &self,
        num_strata_deposited: u64,
        retained_ranks: &[u64],
    ) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let keep = compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        );
        retained_ranks
            .iter()
            .copied()
            .filter(|rank| !keep.contains(rank))
            .collect()
    }

    fn iter_retained_ranks(
        &self,
        num_strata_deposited: u64,
    ) -> Box<dyn Iterator<Item = u64> + '_> {
        let ranks = compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        );
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        )
        .len() as u64
    }

    fn calc_rank_at_column_index(
        &self,
        index: usize,
        num_strata_deposited: u64,
    ) -> u64 {
        let ranks = compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        );
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        let ranks = compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        );
        if ranks.len() <= 1 {
            return 0;
        }
        ranks
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    fn algo_identifier(&self) -> &'static str {
        "geom_seq_nth_root_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_small_depositions() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        // 4 depositions (degree=3, so we retain first 3 + newest)
        let ranks: Vec<u64> = policy.iter_retained_ranks(4).collect();
        assert!(ranks.contains(&0));
        assert!(ranks.contains(&3)); // newest
    }

    #[test]
    fn test_endpoints_always_retained() {
        let policy = GeometricSeqNthRootPolicy::new(4, 2);
        for n in 1..100 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            assert_eq!(*ranks.first().unwrap(), 0, "rank 0 missing at n={n}");
            assert_eq!(
                *ranks.last().unwrap(),
                n - 1,
                "newest rank missing at n={n}"
            );
        }
    }

    #[test]
    fn test_retained_count_consistency() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        for n in 0..100 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_ranks_are_sorted() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        for n in 0..200 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            for w in ranks.windows(2) {
                assert!(w[0] < w[1], "not sorted at n={n}: {:?}", ranks);
            }
        }
    }

    #[test]
    fn test_with_degree_constructor() {
        let p = GeometricSeqNthRootPolicy::with_degree(5);
        assert_eq!(p.degree, 5);
        assert_eq!(p.interspersal, 2);
    }

    #[test]
    fn test_gen_drop_ranks_correct_set() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        let retained = compute_retained_ranks(3, 2, 50);
        let dropped = policy.gen_drop_ranks(50, &retained);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(50).collect();
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(policy.calc_rank_at_column_index(i, 50), r);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        assert_eq!(policy.algo_identifier(), "geom_seq_nth_root_algo");
    }

    #[test]
    fn test_first_degree_strata_retained() {
        let policy = GeometricSeqNthRootPolicy::new(5, 2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(100).collect();
        // First `degree` ranks should be retained
        for r in 0..5 {
            assert!(ranks.contains(&r), "rank {r} should be retained");
        }
    }

    #[test]
    fn test_large_depositions() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1000).collect();
        assert!(!ranks.is_empty());
        assert_eq!(*ranks.first().unwrap(), 0);
        assert_eq!(*ranks.last().unwrap(), 999);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeometricSeqNthRootPolicy>();
    }
}

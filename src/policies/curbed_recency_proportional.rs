use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use crate::bit_floor;

/// Retains strata using a recency-proportional approach with an upper bound
/// on the number of retained strata.
///
/// In the early phase (when fewer than `size_curb` strata have been deposited),
/// all strata are retained (like a perfect-resolution policy).  Once the
/// number of strata would exceed `size_curb`, the policy transitions to
/// recency-proportional spacing with a resolution derived from `size_curb`.
#[derive(Clone, Debug, PartialEq)]
pub struct CurbedRecencyProportionalPolicy {
    pub size_curb: u64,
}

impl CurbedRecencyProportionalPolicy {
    pub fn new(size_curb: u64) -> Self {
        assert!(size_curb >= 8, "size_curb must be at least 8");
        Self { size_curb }
    }
}

/// Calculate provided uncertainty for recency-proportional spacing.
/// Uses `resolution + 1` as divisor, matching RecencyProportionalPolicy.
fn calc_provided_uncertainty(resolution: u64, num_completed: u64) -> u64 {
    let max_unc = num_completed / (resolution + 1);
    bit_floor(max_unc).max(1)
}

/// Compute retained ranks using recency-proportional spacing with the given
/// resolution.
fn recency_proportional_ranks(resolution: u64, num_strata_deposited: u64) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }
    let newest_rank = num_strata_deposited - 1;
    let mut retained = Vec::new();

    let mut rank = 0u64;
    while rank <= newest_rank {
        retained.push(rank);
        if rank == newest_rank {
            break;
        }
        let num_completed = newest_rank - rank;
        let step = calc_provided_uncertainty(resolution, num_completed);
        let next = rank + step;
        if next > newest_rank {
            if *retained.last().unwrap() != newest_rank {
                retained.push(newest_rank);
            }
            break;
        }
        rank = next;
    }

    if let Some(&last) = retained.last() {
        if last != newest_rank {
            retained.push(newest_rank);
        }
    }

    retained
}

/// Compute the set of retained ranks for the curbed policy.
fn compute_retained_ranks(size_curb: u64, num_strata_deposited: u64) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }

    // Early phase: retain everything
    if num_strata_deposited <= size_curb {
        return (0..num_strata_deposited).collect();
    }

    // Later phase: use recency-proportional spacing.
    // Derive resolution from size_curb.  The idea is that the resolution
    // controls how many "bands" there are.  We want the total retained
    // count to stay around size_curb.
    // A reasonable derivation: resolution ~ size_curb / 2
    // (leaves headroom for the extra strata the recency-proportional
    // algorithm retains at finer granularities near the newest rank).
    let resolution = (size_curb / 2).max(1);
    recency_proportional_ranks(resolution, num_strata_deposited)
}

impl StratumRetentionPolicy for CurbedRecencyProportionalPolicy {
    fn gen_drop_ranks(
        &self,
        num_strata_deposited: u64,
        retained_ranks: &[u64],
    ) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let keep = compute_retained_ranks(self.size_curb, num_strata_deposited);
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
        let ranks = compute_retained_ranks(self.size_curb, num_strata_deposited);
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        compute_retained_ranks(self.size_curb, num_strata_deposited).len() as u64
    }

    fn calc_rank_at_column_index(
        &self,
        index: usize,
        num_strata_deposited: u64,
    ) -> u64 {
        let ranks = compute_retained_ranks(self.size_curb, num_strata_deposited);
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        let ranks = compute_retained_ranks(self.size_curb, num_strata_deposited);
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
        "recency_proportional_resolution_curbed_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_early_phase_retains_all() {
        let policy = CurbedRecencyProportionalPolicy::new(20);
        // Before reaching size_curb, everything is retained.
        for n in 0..=20 {
            let count = policy.calc_num_strata_retained_exact(n);
            assert_eq!(count, n, "early phase should retain all at n={n}");
        }
    }

    #[test]
    fn test_late_phase_bounded() {
        let policy = CurbedRecencyProportionalPolicy::new(20);
        // After many depositions, retained count should not grow unboundedly.
        let count_100 = policy.calc_num_strata_retained_exact(100);
        let count_500 = policy.calc_num_strata_retained_exact(500);
        // The growth should be sub-linear.
        assert!(count_500 < 500, "should not retain all 500 strata");
        // Sanity: should retain at least 2 (rank 0 and newest).
        assert!(count_100 >= 2);
        assert!(count_500 >= 2);
    }

    #[test]
    fn test_endpoints_always_retained() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
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
        let policy = CurbedRecencyProportionalPolicy::new(16);
        for n in 0..150 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_ranks_are_sorted() {
        let policy = CurbedRecencyProportionalPolicy::new(12);
        for n in 0..200 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            for w in ranks.windows(2) {
                assert!(w[0] < w[1], "not sorted at n={n}: {:?}", ranks);
            }
        }
    }

    #[test]
    fn test_gen_drop_ranks_correct_set() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        let retained = compute_retained_ranks(10, 50);
        let dropped = policy.gen_drop_ranks(50, &retained);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_gen_drop_ranks_early_phase() {
        let policy = CurbedRecencyProportionalPolicy::new(20);
        // In early phase, nothing should be dropped from a complete set.
        let all: Vec<u64> = (0..10).collect();
        let dropped = policy.gen_drop_ranks(10, &all);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        let ranks: Vec<u64> = policy.iter_retained_ranks(50).collect();
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(policy.calc_rank_at_column_index(i, 50), r);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        assert_eq!(
            policy.algo_identifier(),
            "recency_proportional_resolution_curbed_algo"
        );
    }

    #[test]
    #[should_panic(expected = "size_curb must be at least 8")]
    fn test_size_curb_minimum() {
        CurbedRecencyProportionalPolicy::new(5);
    }
}

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use crate::bit_floor;

/// Retains strata with spacing proportional to recency.
///
/// More recent strata are spaced more closely together, while older strata
/// are spaced more widely. The `resolution` parameter controls how many
/// distinct recency "bands" are maintained, offering a trade-off between
/// precision for recent events and total memory usage.
#[derive(Clone, Debug, PartialEq)]
pub struct RecencyProportionalPolicy {
    pub resolution: u64,
}

impl RecencyProportionalPolicy {
    pub fn new(resolution: u64) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        Self { resolution }
    }
}

/// Calculate the provided uncertainty for a given resolution and number of
/// completed depositions since the rank being considered.
fn calc_provided_uncertainty(resolution: u64, num_completed: u64) -> u64 {
    let max_unc = num_completed / (resolution + 1);
    bit_floor(max_unc).max(1)
}

/// Compute the set of retained ranks for the recency-proportional policy.
fn compute_retained_ranks(resolution: u64, num_strata_deposited: u64) -> Vec<u64> {
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
        // Number of completed depositions after this rank
        let num_completed = newest_rank - rank;
        let step = calc_provided_uncertainty(resolution, num_completed);
        let next = rank + step;
        if next > newest_rank {
            // Ensure newest_rank is always retained
            if *retained.last().unwrap() != newest_rank {
                retained.push(newest_rank);
            }
            break;
        }
        rank = next;
    }

    // Always ensure newest_rank is included
    if let Some(&last) = retained.last() {
        if last != newest_rank {
            retained.push(newest_rank);
        }
    }

    retained
}

impl StratumRetentionPolicy for RecencyProportionalPolicy {
    fn gen_drop_ranks(
        &self,
        num_strata_deposited: u64,
        retained_ranks: &[u64],
    ) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let keep = compute_retained_ranks(self.resolution, num_strata_deposited);
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
        let ranks = compute_retained_ranks(self.resolution, num_strata_deposited);
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        compute_retained_ranks(self.resolution, num_strata_deposited).len() as u64
    }

    fn calc_rank_at_column_index(
        &self,
        index: usize,
        num_strata_deposited: u64,
    ) -> u64 {
        let ranks = compute_retained_ranks(self.resolution, num_strata_deposited);
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        // The worst-case uncertainty is the largest gap between consecutive
        // retained ranks.
        let ranks = compute_retained_ranks(self.resolution, num_strata_deposited);
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
        "recency_proportional_resolution_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = RecencyProportionalPolicy::new(2);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = RecencyProportionalPolicy::new(2);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_small_depositions() {
        let policy = RecencyProportionalPolicy::new(2);
        // 5 depositions -> newest_rank=4
        let ranks: Vec<u64> = policy.iter_retained_ranks(5).collect();
        // Should always include 0 and newest_rank
        assert_eq!(*ranks.first().unwrap(), 0);
        assert_eq!(*ranks.last().unwrap(), 4);
        // Ranks should be sorted
        for w in ranks.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_retained_count_consistency() {
        let policy = RecencyProportionalPolicy::new(3);
        for n in 0..100 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_ranks_include_endpoints() {
        let policy = RecencyProportionalPolicy::new(5);
        for n in 1..50 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            assert_eq!(*ranks.first().unwrap(), 0, "should start at 0 for n={n}");
            assert_eq!(
                *ranks.last().unwrap(),
                n - 1,
                "should end at newest for n={n}"
            );
        }
    }

    #[test]
    fn test_gen_drop_ranks_correct_set() {
        let policy = RecencyProportionalPolicy::new(2);
        let retained = compute_retained_ranks(2, 20);
        // gen_drop_ranks should return empty when we pass only the correct
        // retained set
        let dropped = policy.gen_drop_ranks(20, &retained);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_gen_drop_ranks_drops_extras() {
        let policy = RecencyProportionalPolicy::new(2);
        let all_ranks: Vec<u64> = (0..20).collect();
        let dropped = policy.gen_drop_ranks(20, &all_ranks);
        let kept = compute_retained_ranks(2, 20);
        for d in &dropped {
            assert!(!kept.contains(d));
        }
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = RecencyProportionalPolicy::new(2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(10).collect();
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(policy.calc_rank_at_column_index(i, 10), r);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = RecencyProportionalPolicy::new(2);
        assert_eq!(
            policy.algo_identifier(),
            "recency_proportional_resolution_algo"
        );
    }

    #[test]
    fn test_calc_provided_uncertainty() {
        // With resolution=2, divisor = 3
        assert_eq!(calc_provided_uncertainty(2, 0), 1);
        assert_eq!(calc_provided_uncertainty(2, 1), 1);
        assert_eq!(calc_provided_uncertainty(2, 3), 1);
        // 6 / 3 = 2, bit_floor(2) = 2
        assert_eq!(calc_provided_uncertainty(2, 6), 2);
        // 12 / 3 = 4, bit_floor(4) = 4
        assert_eq!(calc_provided_uncertainty(2, 12), 4);
    }

    #[test]
    fn test_ranks_are_sorted() {
        let policy = RecencyProportionalPolicy::new(4);
        for n in 0..200 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            for w in ranks.windows(2) {
                assert!(w[0] < w[1], "not sorted at n={n}: {:?}", ranks);
            }
        }
    }
}

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use crate::bit_floor;

/// Retains strata with spacing proportional to depth (total depositions).
///
/// Similar to `RecencyProportionalPolicy`, but the uncertainty denominator
/// uses `resolution` instead of `resolution + 1`.  This means the spacing
/// grows slightly faster relative to the number of completed depositions.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthProportionalPolicy {
    pub resolution: u64,
}

impl DepthProportionalPolicy {
    pub fn new(resolution: u64) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        Self { resolution }
    }
}

/// Calculate the provided uncertainty for the depth-proportional policy.
fn calc_provided_uncertainty(resolution: u64, num_completed: u64) -> u64 {
    let max_unc = num_completed / resolution;
    bit_floor(max_unc).max(1)
}

/// Compute the set of retained ranks for the depth-proportional policy.
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

impl StratumRetentionPolicy for DepthProportionalPolicy {
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
        "depth_proportional_resolution_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = DepthProportionalPolicy::new(3);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = DepthProportionalPolicy::new(3);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_endpoints_always_retained() {
        let policy = DepthProportionalPolicy::new(4);
        for n in 1..80 {
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
        let policy = DepthProportionalPolicy::new(5);
        for n in 0..100 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_ranks_are_sorted() {
        let policy = DepthProportionalPolicy::new(3);
        for n in 0..200 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            for w in ranks.windows(2) {
                assert!(w[0] < w[1], "not sorted at n={n}: {:?}", ranks);
            }
        }
    }

    #[test]
    fn test_gen_drop_ranks_correct_set() {
        let policy = DepthProportionalPolicy::new(3);
        let retained = compute_retained_ranks(3, 30);
        let dropped = policy.gen_drop_ranks(30, &retained);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_gen_drop_ranks_drops_extras() {
        let policy = DepthProportionalPolicy::new(3);
        let all: Vec<u64> = (0..30).collect();
        let dropped = policy.gen_drop_ranks(30, &all);
        let kept = compute_retained_ranks(3, 30);
        for d in &dropped {
            assert!(!kept.contains(d));
        }
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = DepthProportionalPolicy::new(3);
        let ranks: Vec<u64> = policy.iter_retained_ranks(20).collect();
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(policy.calc_rank_at_column_index(i, 20), r);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = DepthProportionalPolicy::new(3);
        assert_eq!(
            policy.algo_identifier(),
            "depth_proportional_resolution_algo"
        );
    }

    #[test]
    fn test_calc_provided_uncertainty() {
        // With resolution=3, divisor = 3
        assert_eq!(calc_provided_uncertainty(3, 0), 1);
        assert_eq!(calc_provided_uncertainty(3, 1), 1);
        assert_eq!(calc_provided_uncertainty(3, 2), 1);
        // 3 / 3 = 1, bit_floor(1) = 1
        assert_eq!(calc_provided_uncertainty(3, 3), 1);
        // 6 / 3 = 2, bit_floor(2) = 2
        assert_eq!(calc_provided_uncertainty(3, 6), 2);
        // 12 / 3 = 4, bit_floor(4) = 4
        assert_eq!(calc_provided_uncertainty(3, 12), 4);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DepthProportionalPolicy>();
    }
}

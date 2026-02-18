use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use crate::bit_floor;

/// Depth-proportional retention with tapered (gradual) purging.
///
/// Uses two spacing stages: the current stage uncertainty and the previous
/// (half) stage uncertainty, with a threshold determining where to transition.
/// Before the first purge boundary, all ranks are retained.
///
/// Matches Python hstrat's `depth_proportional_resolution_tapered_algo`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthProportionalTaperedPolicy {
    pub resolution: u64,
}

impl DepthProportionalTaperedPolicy {
    pub fn new(resolution: u64) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        Self { resolution }
    }
}

/// Calculate the provided uncertainty for the depth-proportional policy.
/// Uses `num_strata_deposited / resolution`, rounded down to a power of 2.
fn calc_provided_uncertainty(resolution: u64, num_strata_deposited: u64) -> u64 {
    let max_unc = num_strata_deposited / resolution;
    bit_floor(max_unc).max(1)
}

/// Compute the target set of retained ranks for the tapered variant.
///
/// Matches Python hstrat's `_IterRetainedRanks` for the tapered algo:
/// - Before first purge boundary: all ranks retained
/// - After: two-stage spacing with cur_stage and prev_stage uncertainties
fn compute_retained_ranks(resolution: u64, num_strata_deposited: u64) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }

    let guaranteed_resolution = resolution;

    // Before first ranks are condemned, use identity mapping
    if num_strata_deposited < guaranteed_resolution * 2 + 1 {
        return (0..num_strata_deposited).collect();
    }

    let cur_stage_uncertainty =
        calc_provided_uncertainty(guaranteed_resolution, num_strata_deposited);

    let prev_stage_uncertainty = cur_stage_uncertainty / 2;
    let prev_stage_max_idx = (num_strata_deposited - 2) / prev_stage_uncertainty;
    // Use i64 arithmetic because the intermediate value can be negative
    // (Python handles this with its arbitrary-precision signed integers)
    let thresh_idx_signed: i64 =
        (2 * prev_stage_max_idx as i64 - 4 * guaranteed_resolution as i64 + 2) / 2;
    // Python's // operator floors towards negative infinity, so negative
    // values floor down. Since we clamp to 0 anyway (thresh_idx is used as
    // an index), this is equivalent.
    let thresh_idx = if thresh_idx_signed < 0 {
        0u64
    } else {
        thresh_idx_signed as u64
    };

    let mut retained = Vec::new();

    // First stage: cur_stage_uncertainty spacing
    let first_range_end = thresh_idx * cur_stage_uncertainty;
    let mut r = 0u64;
    while r < first_range_end {
        retained.push(r);
        r += cur_stage_uncertainty;
    }

    // Second stage: prev_stage_uncertainty spacing
    r = thresh_idx * cur_stage_uncertainty;
    while r < num_strata_deposited {
        retained.push(r);
        r += prev_stage_uncertainty;
    }

    // Possibly append last_rank
    let last_rank = num_strata_deposited - 1;
    if thresh_idx * cur_stage_uncertainty > last_rank {
        if last_rank > 0 && !last_rank.is_multiple_of(cur_stage_uncertainty) {
            retained.push(last_rank);
        }
    } else if last_rank > 0 && !last_rank.is_multiple_of(prev_stage_uncertainty) {
        retained.push(last_rank);
    }

    // Deduplicate and sort (the two ranges might overlap at the boundary)
    retained.sort_unstable();
    retained.dedup();
    retained
}

impl StratumRetentionPolicy for DepthProportionalTaperedPolicy {
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64> {
        if num_strata_deposited <= 1 {
            return Vec::new();
        }

        let target_set = compute_retained_ranks(self.resolution, num_strata_deposited);

        // Find strata that are not in the target set.
        let mut non_conforming: Vec<u64> = retained_ranks
            .iter()
            .copied()
            .filter(|r| target_set.binary_search(r).is_err())
            .collect();

        // Sort ascending so oldest is first.
        non_conforming.sort_unstable();

        // Tapered: only drop the oldest non-conforming stratum.
        if let Some(&oldest) = non_conforming.first() {
            alloc::vec![oldest]
        } else {
            Vec::new()
        }
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        let ranks = compute_retained_ranks(self.resolution, num_strata_deposited);
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        compute_retained_ranks(self.resolution, num_strata_deposited).len() as u64
    }

    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64 {
        let ranks: Vec<u64> = self.iter_retained_ranks(num_strata_deposited).collect();
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
        ranks.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }

    fn algo_identifier(&self) -> &'static str {
        "depth_proportional_resolution_tapered_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deposited() {
        let policy = DepthProportionalTaperedPolicy { resolution: 10 };
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        assert_eq!(policy.gen_drop_ranks(0, &[]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(0).collect::<Vec<_>>(),
            Vec::<u64>::new(),
        );
    }

    #[test]
    fn test_one_deposited() {
        let policy = DepthProportionalTaperedPolicy { resolution: 10 };
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        assert_eq!(policy.gen_drop_ranks(1, &[0]), Vec::<u64>::new());
        assert_eq!(policy.iter_retained_ranks(1).collect::<Vec<_>>(), vec![0],);
    }

    #[test]
    fn test_small_deposited() {
        let policy = DepthProportionalTaperedPolicy { resolution: 10 };
        // With 5 strata deposited, 5 < 10*2+1 = 21, so identity mapping
        assert_eq!(
            policy.iter_retained_ranks(5).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4],
        );
        assert_eq!(
            policy.gen_drop_ranks(5, &[0, 1, 2, 3, 4]),
            Vec::<u64>::new(),
        );
    }

    #[test]
    fn test_algo_identifier() {
        let policy = DepthProportionalTaperedPolicy { resolution: 10 };
        assert_eq!(
            policy.algo_identifier(),
            "depth_proportional_resolution_tapered_algo",
        );
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DepthProportionalTaperedPolicy>();
    }

    #[test]
    fn test_always_retains_first_and_last() {
        let policy = DepthProportionalTaperedPolicy { resolution: 3 };
        for n in 1..200u64 {
            let retained: Vec<u64> = policy.iter_retained_ranks(n).collect();
            assert!(retained.contains(&0), "n={}: should retain rank 0", n,);
            assert!(
                retained.contains(&(n - 1)),
                "n={}: should retain newest rank {}",
                n,
                n - 1,
            );
        }
    }
}

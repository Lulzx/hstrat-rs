use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Retains strata at fixed rank intervals.
///
/// Keeps every `resolution`-th rank (0, resolution, 2*resolution, ...)
/// plus the newest rank. Provides constant absolute MRCA uncertainty
/// equal to `resolution`.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedResolutionPolicy {
    pub resolution: u64,
}

impl FixedResolutionPolicy {
    pub fn new(resolution: u64) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        Self { resolution }
    }
}

impl StratumRetentionPolicy for FixedResolutionPolicy {
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let newest_rank = num_strata_deposited - 1;
        retained_ranks
            .iter()
            .copied()
            .filter(|&rank| rank % self.resolution != 0 && rank != newest_rank)
            .collect()
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        if num_strata_deposited == 0 {
            return Box::new(core::iter::empty());
        }
        let newest_rank = num_strata_deposited - 1;
        let resolution = self.resolution;
        let aligned_iter = (0..=newest_rank).step_by(resolution as usize);
        // If newest_rank is not aligned, append it
        let needs_extra = !newest_rank.is_multiple_of(resolution);
        Box::new(aligned_iter.chain(if needs_extra { Some(newest_rank) } else { None }))
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited == 0 {
            return 0;
        }
        let newest_rank = num_strata_deposited - 1;
        let aligned_count = newest_rank / self.resolution + 1;
        let extra = if !newest_rank.is_multiple_of(self.resolution) {
            1
        } else {
            0
        };
        aligned_count + extra
    }

    fn fast_drop_singleton_is_second_most_recent(&self) -> bool {
        true
    }

    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64 {
        let ranks: Vec<u64> = self.iter_retained_ranks(num_strata_deposited).collect();
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, _num_strata_deposited: u64) -> u64 {
        self.resolution
    }

    fn algo_identifier(&self) -> &'static str {
        "fixed_resolution_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = FixedResolutionPolicy::new(10);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
        assert!(policy.gen_drop_ranks(0, &[]).is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = FixedResolutionPolicy::new(10);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_aligned_depositions() {
        let policy = FixedResolutionPolicy::new(5);
        // 11 depositions -> newest_rank = 10
        // retained: 0, 5, 10
        assert_eq!(policy.calc_num_strata_retained_exact(11), 3);
        let ranks: Vec<u64> = policy.iter_retained_ranks(11).collect();
        assert_eq!(ranks, vec![0, 5, 10]);
    }

    #[test]
    fn test_unaligned_depositions() {
        let policy = FixedResolutionPolicy::new(5);
        // 8 depositions -> newest_rank = 7
        // retained: 0, 5, 7
        assert_eq!(policy.calc_num_strata_retained_exact(8), 3);
        let ranks: Vec<u64> = policy.iter_retained_ranks(8).collect();
        assert_eq!(ranks, vec![0, 5, 7]);
    }

    #[test]
    fn test_resolution_one() {
        let policy = FixedResolutionPolicy::new(1);
        // Resolution 1 retains every rank
        assert_eq!(policy.calc_num_strata_retained_exact(5), 5);
        let ranks: Vec<u64> = policy.iter_retained_ranks(5).collect();
        assert_eq!(ranks, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_gen_drop_ranks() {
        let policy = FixedResolutionPolicy::new(5);
        // After 8 depositions (newest_rank=7), retained ranks should be
        // 0, 5, 7. Anything else should be dropped.
        let retained = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let dropped = policy.gen_drop_ranks(8, &retained);
        assert_eq!(dropped, vec![1, 2, 3, 4, 6]);
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = FixedResolutionPolicy::new(5);
        assert_eq!(policy.calc_rank_at_column_index(0, 8), 0);
        assert_eq!(policy.calc_rank_at_column_index(1, 8), 5);
        assert_eq!(policy.calc_rank_at_column_index(2, 8), 7);
    }

    #[test]
    fn test_mrca_uncertainty() {
        let policy = FixedResolutionPolicy::new(10);
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(100), 10);
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(1), 10);
    }

    #[test]
    fn test_algo_identifier() {
        let policy = FixedResolutionPolicy::new(5);
        assert_eq!(policy.algo_identifier(), "fixed_resolution_algo");
    }

    #[test]
    fn test_num_retained_equals_iter_count() {
        let policy = FixedResolutionPolicy::new(7);
        for n in 0..100 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }
}

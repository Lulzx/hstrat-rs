use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Retains only the first and most recent strata -- minimal memory usage,
/// maximum MRCA uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct NominalResolutionPolicy;

impl NominalResolutionPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NominalResolutionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl StratumRetentionPolicy for NominalResolutionPolicy {
    fn gen_drop_ranks(
        &self,
        num_strata_deposited: u64,
        retained_ranks: &[u64],
    ) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let newest_rank = num_strata_deposited - 1;
        retained_ranks
            .iter()
            .copied()
            .filter(|&rank| rank != 0 && rank != newest_rank)
            .collect()
    }

    fn iter_retained_ranks(
        &self,
        num_strata_deposited: u64,
    ) -> Box<dyn Iterator<Item = u64> + '_> {
        if num_strata_deposited == 0 {
            Box::new(core::iter::empty())
        } else if num_strata_deposited == 1 {
            Box::new(core::iter::once(0u64))
        } else {
            Box::new([0u64, num_strata_deposited - 1].into_iter())
        }
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        match num_strata_deposited {
            0 => 0,
            1 => 1,
            _ => 2,
        }
    }

    fn calc_rank_at_column_index(
        &self,
        index: usize,
        num_strata_deposited: u64,
    ) -> u64 {
        let ranks: Vec<u64> = self.iter_retained_ranks(num_strata_deposited).collect();
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            0
        } else {
            num_strata_deposited - 1
        }
    }

    fn algo_identifier(&self) -> &'static str {
        "nominal_resolution_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deposited() {
        let policy = NominalResolutionPolicy;
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        assert_eq!(policy.gen_drop_ranks(0, &[]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(0).collect::<Vec<_>>(),
            Vec::<u64>::new(),
        );
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(0), 0);
    }

    #[test]
    fn test_one_deposited() {
        let policy = NominalResolutionPolicy;
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        assert_eq!(policy.gen_drop_ranks(1, &[0]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(1).collect::<Vec<_>>(),
            vec![0],
        );
        assert_eq!(policy.calc_rank_at_column_index(0, 1), 0);
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(1), 0);
    }

    #[test]
    fn test_two_deposited() {
        let policy = NominalResolutionPolicy;
        assert_eq!(policy.calc_num_strata_retained_exact(2), 2);
        assert_eq!(policy.gen_drop_ranks(2, &[0, 1]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(2).collect::<Vec<_>>(),
            vec![0, 1],
        );
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(2), 1);
    }

    #[test]
    fn test_many_deposited() {
        let policy = NominalResolutionPolicy;
        let n = 10;
        assert_eq!(policy.calc_num_strata_retained_exact(n), 2);
        assert_eq!(
            policy.iter_retained_ranks(n).collect::<Vec<_>>(),
            vec![0, n - 1],
        );
        assert_eq!(policy.calc_rank_at_column_index(0, n), 0);
        assert_eq!(policy.calc_rank_at_column_index(1, n), n - 1);
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(n), n - 1);
    }

    #[test]
    fn test_gen_drop_ranks_drops_middle() {
        let policy = NominalResolutionPolicy;
        let mut drops = policy.gen_drop_ranks(5, &[0, 1, 2, 3, 4]);
        drops.sort();
        assert_eq!(drops, vec![1, 2, 3]);
    }

    #[test]
    fn test_algo_identifier() {
        let policy = NominalResolutionPolicy;
        assert_eq!(policy.algo_identifier(), "nominal_resolution_algo");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NominalResolutionPolicy>();
    }
}

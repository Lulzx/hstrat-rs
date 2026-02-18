use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Retains every stratum ever deposited -- zero MRCA uncertainty, but O(n)
/// memory usage.
#[derive(Clone, Debug, PartialEq)]
pub struct PerfectResolutionPolicy;

impl PerfectResolutionPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PerfectResolutionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl StratumRetentionPolicy for PerfectResolutionPolicy {
    fn gen_drop_ranks(&self, _num_strata_deposited: u64, _retained_ranks: &[u64]) -> Vec<u64> {
        // Never drop anything.
        Vec::new()
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(0..num_strata_deposited)
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        num_strata_deposited
    }

    fn calc_rank_at_column_index(&self, index: usize, _num_strata_deposited: u64) -> u64 {
        index as u64
    }

    fn calc_mrca_uncertainty_abs_exact(&self, _num_strata_deposited: u64) -> u64 {
        0
    }

    fn algo_identifier(&self) -> &'static str {
        "perfect_resolution_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deposited() {
        let policy = PerfectResolutionPolicy;
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        assert_eq!(policy.gen_drop_ranks(0, &[]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(0).collect::<Vec<_>>(),
            Vec::<u64>::new(),
        );
    }

    #[test]
    fn test_one_deposited() {
        let policy = PerfectResolutionPolicy;
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        assert_eq!(policy.gen_drop_ranks(1, &[0]), Vec::<u64>::new());
        assert_eq!(policy.iter_retained_ranks(1).collect::<Vec<_>>(), vec![0],);
        assert_eq!(policy.calc_rank_at_column_index(0, 1), 0);
    }

    #[test]
    fn test_many_deposited() {
        let policy = PerfectResolutionPolicy;
        let n = 10;
        assert_eq!(policy.calc_num_strata_retained_exact(n), n);
        let retained: Vec<u64> = (0..n).collect();
        assert_eq!(policy.gen_drop_ranks(n, &retained), Vec::<u64>::new());
        assert_eq!(policy.iter_retained_ranks(n).collect::<Vec<_>>(), retained,);
        for i in 0..n as usize {
            assert_eq!(policy.calc_rank_at_column_index(i, n), i as u64);
        }
    }

    #[test]
    fn test_uncertainty_always_zero() {
        let policy = PerfectResolutionPolicy;
        for n in 0..20 {
            assert_eq!(policy.calc_mrca_uncertainty_abs_exact(n), 0);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = PerfectResolutionPolicy;
        assert_eq!(policy.algo_identifier(), "perfect_resolution_algo");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PerfectResolutionPolicy>();
    }
}

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::*;

/// A runtime-dispatched wrapper over all 11 stratum retention policy types.
///
/// `DynamicPolicy` enables selecting a policy at runtime rather than at compile
/// time, which is needed for PyO3 bindings where Python does not have Rust
/// generics available.  Each trait method delegates to the inner variant via
/// a `match` dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicPolicy {
    FixedResolution(FixedResolutionPolicy),
    RecencyProportional(RecencyProportionalPolicy),
    CurbedRecencyProportional(CurbedRecencyProportionalPolicy),
    GeometricSeqNthRoot(GeometricSeqNthRootPolicy),
    GeometricSeqNthRootTapered(GeometricSeqNthRootTaperedPolicy),
    DepthProportional(DepthProportionalPolicy),
    DepthProportionalTapered(DepthProportionalTaperedPolicy),
    NominalResolution(NominalResolutionPolicy),
    PerfectResolution(PerfectResolutionPolicy),
    Pseudostochastic(PseudostochasticPolicy),
    Stochastic(StochasticPolicy),
}

/// Helper macro that dispatches a method call to the inner policy variant.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            DynamicPolicy::FixedResolution(p) => p.$method($($arg),*),
            DynamicPolicy::RecencyProportional(p) => p.$method($($arg),*),
            DynamicPolicy::CurbedRecencyProportional(p) => p.$method($($arg),*),
            DynamicPolicy::GeometricSeqNthRoot(p) => p.$method($($arg),*),
            DynamicPolicy::GeometricSeqNthRootTapered(p) => p.$method($($arg),*),
            DynamicPolicy::DepthProportional(p) => p.$method($($arg),*),
            DynamicPolicy::DepthProportionalTapered(p) => p.$method($($arg),*),
            DynamicPolicy::NominalResolution(p) => p.$method($($arg),*),
            DynamicPolicy::PerfectResolution(p) => p.$method($($arg),*),
            DynamicPolicy::Pseudostochastic(p) => p.$method($($arg),*),
            DynamicPolicy::Stochastic(p) => p.$method($($arg),*),
        }
    };
}

impl StratumRetentionPolicy for DynamicPolicy {
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64> {
        dispatch!(self, gen_drop_ranks, num_strata_deposited, retained_ranks)
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        dispatch!(self, iter_retained_ranks, num_strata_deposited)
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        dispatch!(self, calc_num_strata_retained_exact, num_strata_deposited)
    }

    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64 {
        dispatch!(self, calc_rank_at_column_index, index, num_strata_deposited)
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        dispatch!(self, calc_mrca_uncertainty_abs_exact, num_strata_deposited)
    }

    fn algo_identifier(&self) -> &'static str {
        dispatch!(self, algo_identifier)
    }
}

// Convenience From impls for each variant.
impl From<FixedResolutionPolicy> for DynamicPolicy {
    fn from(p: FixedResolutionPolicy) -> Self {
        DynamicPolicy::FixedResolution(p)
    }
}

impl From<RecencyProportionalPolicy> for DynamicPolicy {
    fn from(p: RecencyProportionalPolicy) -> Self {
        DynamicPolicy::RecencyProportional(p)
    }
}

impl From<CurbedRecencyProportionalPolicy> for DynamicPolicy {
    fn from(p: CurbedRecencyProportionalPolicy) -> Self {
        DynamicPolicy::CurbedRecencyProportional(p)
    }
}

impl From<GeometricSeqNthRootPolicy> for DynamicPolicy {
    fn from(p: GeometricSeqNthRootPolicy) -> Self {
        DynamicPolicy::GeometricSeqNthRoot(p)
    }
}

impl From<GeometricSeqNthRootTaperedPolicy> for DynamicPolicy {
    fn from(p: GeometricSeqNthRootTaperedPolicy) -> Self {
        DynamicPolicy::GeometricSeqNthRootTapered(p)
    }
}

impl From<DepthProportionalPolicy> for DynamicPolicy {
    fn from(p: DepthProportionalPolicy) -> Self {
        DynamicPolicy::DepthProportional(p)
    }
}

impl From<DepthProportionalTaperedPolicy> for DynamicPolicy {
    fn from(p: DepthProportionalTaperedPolicy) -> Self {
        DynamicPolicy::DepthProportionalTapered(p)
    }
}

impl From<NominalResolutionPolicy> for DynamicPolicy {
    fn from(p: NominalResolutionPolicy) -> Self {
        DynamicPolicy::NominalResolution(p)
    }
}

impl From<PerfectResolutionPolicy> for DynamicPolicy {
    fn from(p: PerfectResolutionPolicy) -> Self {
        DynamicPolicy::PerfectResolution(p)
    }
}

impl From<PseudostochasticPolicy> for DynamicPolicy {
    fn from(p: PseudostochasticPolicy) -> Self {
        DynamicPolicy::Pseudostochastic(p)
    }
}

impl From<StochasticPolicy> for DynamicPolicy {
    fn from(p: StochasticPolicy) -> Self {
        DynamicPolicy::Stochastic(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_perfect_resolution() {
        let dp = DynamicPolicy::PerfectResolution(PerfectResolutionPolicy::new());
        assert_eq!(dp.calc_num_strata_retained_exact(10), 10);
        assert_eq!(dp.calc_mrca_uncertainty_abs_exact(10), 0);
        assert_eq!(dp.algo_identifier(), "perfect_resolution_algo");
        let ranks: Vec<u64> = dp.iter_retained_ranks(5).collect();
        assert_eq!(ranks, vec![0, 1, 2, 3, 4]);
        assert!(dp.gen_drop_ranks(5, &[0, 1, 2, 3, 4]).is_empty());
    }

    #[test]
    fn test_dynamic_fixed_resolution() {
        let dp = DynamicPolicy::FixedResolution(FixedResolutionPolicy::new(5));
        // 11 depositions -> newest_rank = 10
        // retained: 0, 5, 10
        assert_eq!(dp.calc_num_strata_retained_exact(11), 3);
        let ranks: Vec<u64> = dp.iter_retained_ranks(11).collect();
        assert_eq!(ranks, vec![0, 5, 10]);
        assert_eq!(dp.calc_rank_at_column_index(1, 11), 5);
        assert_eq!(dp.calc_mrca_uncertainty_abs_exact(11), 5);
        assert_eq!(dp.algo_identifier(), "fixed_resolution_algo");
    }

    #[test]
    fn test_dynamic_nominal_resolution() {
        let dp = DynamicPolicy::NominalResolution(NominalResolutionPolicy::new());
        assert_eq!(dp.calc_num_strata_retained_exact(0), 0);
        assert_eq!(dp.calc_num_strata_retained_exact(1), 1);
        assert_eq!(dp.calc_num_strata_retained_exact(100), 2);
        let ranks: Vec<u64> = dp.iter_retained_ranks(100).collect();
        assert_eq!(ranks, vec![0, 99]);
    }

    #[test]
    fn test_dynamic_from_impls() {
        let dp: DynamicPolicy = FixedResolutionPolicy::new(3).into();
        assert_eq!(dp.algo_identifier(), "fixed_resolution_algo");

        let dp: DynamicPolicy = PerfectResolutionPolicy::new().into();
        assert_eq!(dp.algo_identifier(), "perfect_resolution_algo");

        let dp: DynamicPolicy = NominalResolutionPolicy::new().into();
        assert_eq!(dp.algo_identifier(), "nominal_resolution_algo");

        let dp: DynamicPolicy = RecencyProportionalPolicy::new(3).into();
        assert_eq!(dp.algo_identifier(), "recency_proportional_resolution_algo");

        let dp: DynamicPolicy = CurbedRecencyProportionalPolicy::new(10).into();
        assert_eq!(
            dp.algo_identifier(),
            "recency_proportional_resolution_curbed_algo"
        );

        let dp: DynamicPolicy = GeometricSeqNthRootPolicy::new(2, 2).into();
        assert_eq!(dp.algo_identifier(), "geom_seq_nth_root_algo");

        let dp: DynamicPolicy = GeometricSeqNthRootTaperedPolicy::new(2, 2).into();
        assert_eq!(dp.algo_identifier(), "geom_seq_nth_root_tapered_algo");

        let dp: DynamicPolicy = DepthProportionalPolicy::new(5).into();
        assert_eq!(dp.algo_identifier(), "depth_proportional_resolution_algo");

        let dp: DynamicPolicy = DepthProportionalTaperedPolicy::new(5).into();
        assert_eq!(
            dp.algo_identifier(),
            "depth_proportional_resolution_tapered_algo"
        );

        let dp: DynamicPolicy = PseudostochasticPolicy::new(10).into();
        assert_eq!(dp.algo_identifier(), "pseudostochastic_algo");

        let dp: DynamicPolicy = StochasticPolicy::new(0.5).into();
        assert_eq!(dp.algo_identifier(), "stochastic_algo");
    }

    #[test]
    fn test_dynamic_gen_drop_ranks_delegates() {
        let dp = DynamicPolicy::FixedResolution(FixedResolutionPolicy::new(5));
        let all: Vec<u64> = (0..8).collect();
        let dropped = dp.gen_drop_ranks(8, &all);
        // After 8 depositions (newest=7), retained: 0, 5, 7
        assert_eq!(dropped, vec![1, 2, 3, 4, 6]);
    }

    #[test]
    fn test_dynamic_clone_and_eq() {
        let dp1 = DynamicPolicy::FixedResolution(FixedResolutionPolicy::new(10));
        let dp2 = dp1.clone();
        assert_eq!(dp1, dp2);

        let dp3 = DynamicPolicy::FixedResolution(FixedResolutionPolicy::new(20));
        assert_ne!(dp1, dp3);

        let dp4 = DynamicPolicy::PerfectResolution(PerfectResolutionPolicy::new());
        assert_ne!(dp1, dp4);
    }
}

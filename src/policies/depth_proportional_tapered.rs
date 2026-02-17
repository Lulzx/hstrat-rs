use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use crate::bit_floor;

/// Depth-proportional retention with tapered (gradual) purging.
///
/// Same target set of retained ranks as the non-tapered depth-proportional
/// policy, but instead of dropping all non-conforming strata at once when a
/// purge boundary is crossed, it drops only the oldest non-conforming stratum
/// per deposition step. This smooths out memory usage spikes.
///
/// The provided uncertainty is `bit_floor(num_completed / resolution).max(1)`,
/// where `num_completed = num_strata_deposited - 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthProportionalTaperedPolicy {
    pub resolution: u64,
}

impl DepthProportionalTaperedPolicy {
    pub fn new(resolution: u64) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        Self { resolution }
    }

    /// Calculate the spacing (step size) between retained strata at the
    /// current depth.
    fn calc_provided_uncertainty(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 1;
        }
        let num_completed = num_strata_deposited - 1;
        let raw = num_completed / self.resolution;
        bit_floor(raw).max(1)
    }

    /// Calculate the target set of retained ranks given the spacing.
    /// Retained ranks are: rank 0, every `spacing`-th rank, and the
    /// newest rank.
    fn calc_target_retained_ranks(
        &self,
        num_strata_deposited: u64,
    ) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let newest_rank = num_strata_deposited - 1;
        let spacing = self.calc_provided_uncertainty(num_strata_deposited);

        let mut ranks = Vec::new();
        let mut r = 0u64;
        while r <= newest_rank {
            ranks.push(r);
            r += spacing;
        }
        // Always include the newest rank.
        if ranks.last().copied() != Some(newest_rank) {
            ranks.push(newest_rank);
        }
        ranks
    }
}

impl StratumRetentionPolicy for DepthProportionalTaperedPolicy {
    fn gen_drop_ranks(
        &self,
        num_strata_deposited: u64,
        retained_ranks: &[u64],
    ) -> Vec<u64> {
        if num_strata_deposited <= 1 {
            return Vec::new();
        }

        let target_set =
            self.calc_target_retained_ranks(num_strata_deposited);

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

    fn iter_retained_ranks(
        &self,
        num_strata_deposited: u64,
    ) -> Box<dyn Iterator<Item = u64> + '_> {
        let ranks = self.calc_target_retained_ranks(num_strata_deposited);
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(
        &self,
        num_strata_deposited: u64,
    ) -> u64 {
        self.calc_target_retained_ranks(num_strata_deposited).len() as u64
    }

    fn calc_rank_at_column_index(
        &self,
        index: usize,
        num_strata_deposited: u64,
    ) -> u64 {
        let ranks: Vec<u64> =
            self.iter_retained_ranks(num_strata_deposited).collect();
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(
        &self,
        num_strata_deposited: u64,
    ) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        self.calc_provided_uncertainty(num_strata_deposited)
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
        assert_eq!(
            policy.iter_retained_ranks(1).collect::<Vec<_>>(),
            vec![0],
        );
    }

    #[test]
    fn test_small_deposited() {
        let policy = DepthProportionalTaperedPolicy { resolution: 10 };
        // With 5 strata deposited, num_completed=4,
        // spacing = bit_floor(4/10).max(1) = bit_floor(0).max(1) = 1
        // Target: 0,1,2,3,4 -- all kept.
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
    fn test_spacing_increases() {
        let policy = DepthProportionalTaperedPolicy { resolution: 2 };
        // With 5 deposited: num_completed=4, raw=4/2=2,
        // bit_floor(2)=2, spacing=2.
        // Target: 0, 2, 4
        assert_eq!(
            policy.iter_retained_ranks(5).collect::<Vec<_>>(),
            vec![0, 2, 4],
        );
    }

    #[test]
    fn test_tapered_drops_one_at_a_time() {
        let policy = DepthProportionalTaperedPolicy { resolution: 2 };
        // After 5 deposited with spacing=2, target={0,2,4}
        // If column has [0,1,2,3,4], non-conforming=[1,3]
        // Tapered: only drops oldest non-conforming = 1
        let drops = policy.gen_drop_ranks(5, &[0, 1, 2, 3, 4]);
        assert_eq!(drops, vec![1]);
    }

    #[test]
    fn test_uncertainty() {
        let policy = DepthProportionalTaperedPolicy { resolution: 10 };
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(0), 0);
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(1), 0);
        // n=11 => num_completed=10, raw=10/10=1, bit_floor(1)=1
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(11), 1);
        // n=21 => num_completed=20, raw=20/10=2, bit_floor(2)=2
        assert_eq!(policy.calc_mrca_uncertainty_abs_exact(21), 2);
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
        for n in 1..50u64 {
            let retained: Vec<u64> =
                policy.iter_retained_ranks(n).collect();
            assert!(
                retained.contains(&0),
                "n={}: should retain rank 0",
                n,
            );
            assert!(
                retained.contains(&(n - 1)),
                "n={}: should retain newest rank {}",
                n,
                n - 1,
            );
        }
    }
}

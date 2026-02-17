use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Geometric sequence Nth root retention with tapered (gradual) purging.
///
/// Retains strata at geometrically spaced positions, achieving O(n^(1/k))
/// space complexity where k is the `degree` parameter. The `interspersal`
/// parameter controls how many extra strata are placed between geometric
/// anchor points for smoother coverage.
///
/// Unlike the non-tapered variant, this drops only the single least-needed
/// stratum per deposition step rather than performing bulk purges.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricSeqNthRootTaperedPolicy {
    pub degree: u64,
    pub interspersal: u64,
}

impl Default for GeometricSeqNthRootTaperedPolicy {
    fn default() -> Self {
        Self {
            degree: 2,
            interspersal: 2,
        }
    }
}

impl GeometricSeqNthRootTaperedPolicy {
    pub fn new(degree: u64, interspersal: u64) -> Self {
        assert!(degree > 0, "degree must be positive");
        Self {
            degree,
            interspersal,
        }
    }

    /// Calculate the target set of ranks to retain for a given depth.
    ///
    /// Uses a layered geometric approach: compute anchor points at
    /// geometrically increasing intervals, and intersperse extra strata
    /// between them.
    fn calc_target_retained_ranks(
        &self,
        num_strata_deposited: u64,
    ) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        if num_strata_deposited == 1 {
            return alloc::vec![0];
        }

        let newest_rank = num_strata_deposited - 1;
        let mut retained = Vec::new();

        // Always retain first and last.
        retained.push(0);

        if newest_rank == 0 {
            return retained;
        }

        // Calculate base for geometric spacing.
        let n = newest_rank as f64;
        let base = if self.degree <= 1 {
            n
        } else {
            libm::fmax(libm::pow(n, 1.0 / self.degree as f64), 2.0)
        };

        // Generate geometric anchor points.
        let mut anchors = Vec::new();
        let mut power = 1.0f64;
        loop {
            let anchor = libm::round(power) as u64;
            if anchor > newest_rank {
                break;
            }
            anchors.push(anchor);
            power *= base;
            if power > newest_rank as f64 * 1.1 {
                break;
            }
        }
        anchors.push(newest_rank);
        anchors.sort_unstable();
        anchors.dedup();

        // Intersperse strata between anchor points.
        for window in anchors.windows(2) {
            let lo = window[0];
            let hi = window[1];
            retained.push(lo);
            if self.interspersal > 0 && hi > lo + 1 {
                let gap = hi - lo;
                let step = (gap / (self.interspersal + 1)).max(1);
                let mut r = lo + step;
                let mut count = 0u64;
                while r < hi && count < self.interspersal {
                    retained.push(r);
                    r += step;
                    count += 1;
                }
            }
        }
        // Include last anchor.
        retained.push(newest_rank);

        retained.sort_unstable();
        retained.dedup();
        retained
    }
}

impl StratumRetentionPolicy for GeometricSeqNthRootTaperedPolicy {
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

        // Find strata not in the target set.
        let mut non_conforming: Vec<u64> = retained_ranks
            .iter()
            .copied()
            .filter(|r| target_set.binary_search(r).is_err())
            .collect();

        if non_conforming.is_empty() {
            return Vec::new();
        }

        // Tapered: drop only the oldest non-conforming stratum.
        non_conforming.sort_unstable();
        alloc::vec![non_conforming[0]]
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
        // Worst-case gap between consecutive retained ranks.
        let retained =
            self.calc_target_retained_ranks(num_strata_deposited);
        if retained.len() <= 1 {
            return num_strata_deposited.saturating_sub(1);
        }
        retained
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    fn algo_identifier(&self) -> &'static str {
        "geom_seq_nth_root_tapered_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deposited() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 2,
        };
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        assert_eq!(policy.gen_drop_ranks(0, &[]), Vec::<u64>::new());
    }

    #[test]
    fn test_one_deposited() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 2,
        };
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        assert_eq!(
            policy.iter_retained_ranks(1).collect::<Vec<_>>(),
            vec![0],
        );
    }

    #[test]
    fn test_two_deposited() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 2,
        };
        let retained: Vec<u64> = policy.iter_retained_ranks(2).collect();
        assert!(retained.contains(&0));
        assert!(retained.contains(&1));
    }

    #[test]
    fn test_always_retains_first_and_last() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 3,
            interspersal: 2,
        };
        for n in 1..100u64 {
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

    #[test]
    fn test_tapered_drops_one_at_a_time() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 2,
        };
        let all_ranks: Vec<u64> = (0..50).collect();
        let drops = policy.gen_drop_ranks(50, &all_ranks);
        assert!(
            drops.len() <= 1,
            "tapered should drop at most 1, got {:?}",
            drops,
        );
    }

    #[test]
    fn test_sublinear_growth() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 2,
        };
        let n = 1000;
        let count = policy.calc_num_strata_retained_exact(n);
        assert!(
            count < n / 2,
            "expected sublinear retention, got {} out of {}",
            count,
            n,
        );
    }

    #[test]
    fn test_retained_ranks_sorted_and_unique() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 3,
        };
        for n in 0..100 {
            let retained: Vec<u64> =
                policy.iter_retained_ranks(n).collect();
            for w in retained.windows(2) {
                assert!(
                    w[0] < w[1],
                    "n={}: ranks not strictly ascending: {:?}",
                    n,
                    retained,
                );
            }
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = GeometricSeqNthRootTaperedPolicy {
            degree: 2,
            interspersal: 2,
        };
        assert_eq!(
            policy.algo_identifier(),
            "geom_seq_nth_root_tapered_algo",
        );
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeometricSeqNthRootTaperedPolicy>();
    }

    #[test]
    fn test_default() {
        let policy = GeometricSeqNthRootTaperedPolicy::default();
        assert_eq!(policy.degree, 2);
        assert_eq!(policy.interspersal, 2);
    }
}

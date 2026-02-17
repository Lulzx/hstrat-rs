use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Stochastic retention policy.
///
/// Retains first (rank 0) and last (newest) strata always. The
/// second-most-recent stratum is retained with probability
/// `retention_probability`. All other strata that have already survived
/// are never dropped.
///
/// Since this uses an RNG, a deterministic seed derived from the rank is
/// used internally for reproducibility in `iter_retained_ranks`.
///
/// `PartialEq` is implemented via bitwise f64 comparison to avoid
/// floating-point equality pitfalls.
#[derive(Clone, Debug)]
pub struct StochasticPolicy {
    pub retention_probability: f64,
}

impl Default for StochasticPolicy {
    fn default() -> Self {
        Self {
            retention_probability: 0.5,
        }
    }
}

impl StochasticPolicy {
    pub fn new(retention_probability: f64) -> Self {
        Self {
            retention_probability,
        }
    }
}

impl PartialEq for StochasticPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.retention_probability.to_bits()
            == other.retention_probability.to_bits()
    }
}

impl StochasticPolicy {
    /// Deterministic pseudo-random decision for whether to retain a rank.
    /// Uses a splitmix64-style hash for good distribution.
    #[inline]
    fn should_retain_deterministic(&self, rank: u64) -> bool {
        let mut z = rank.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map to [0, 1).
        let hash_val = (z as f64) / (u64::MAX as f64);
        hash_val < self.retention_probability
    }
}

impl StratumRetentionPolicy for StochasticPolicy {
    fn gen_drop_ranks(
        &self,
        num_strata_deposited: u64,
        retained_ranks: &[u64],
    ) -> Vec<u64> {
        if num_strata_deposited <= 2 {
            return Vec::new();
        }

        let newest_rank = num_strata_deposited - 1;
        let second_most_recent = newest_rank - 1;

        // Only consider dropping the second-most-recent.
        if second_most_recent == 0 {
            return Vec::new();
        }

        if !retained_ranks.contains(&second_most_recent) {
            return Vec::new();
        }

        if self.should_retain_deterministic(second_most_recent) {
            Vec::new()
        } else {
            alloc::vec![second_most_recent]
        }
    }

    fn iter_retained_ranks(
        &self,
        num_strata_deposited: u64,
    ) -> Box<dyn Iterator<Item = u64> + '_> {
        if num_strata_deposited == 0 {
            return Box::new(core::iter::empty());
        }
        if num_strata_deposited == 1 {
            return Box::new(core::iter::once(0));
        }

        let newest_rank = num_strata_deposited - 1;
        let mut retained = Vec::new();
        retained.push(0u64);

        for dep in 1..num_strata_deposited {
            retained.push(dep);

            if dep >= 2 {
                let second_most_recent = dep - 1;
                if second_most_recent != 0
                    && !self.should_retain_deterministic(second_most_recent)
                {
                    retained.retain(|&r| r != second_most_recent);
                }
            }
        }

        if !retained.contains(&0) {
            retained.insert(0, 0);
        }
        if !retained.contains(&newest_rank) {
            retained.push(newest_rank);
        }

        retained.sort_unstable();
        retained.dedup();
        Box::new(retained.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        self.iter_retained_ranks(num_strata_deposited).count() as u64
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

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        let retained: Vec<u64> =
            self.iter_retained_ranks(num_strata_deposited).collect();
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
        "stochastic_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deposited() {
        let policy = StochasticPolicy::default();
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        assert_eq!(policy.gen_drop_ranks(0, &[]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(0).collect::<Vec<_>>(),
            Vec::<u64>::new(),
        );
    }

    #[test]
    fn test_one_deposited() {
        let policy = StochasticPolicy::default();
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        assert_eq!(
            policy.iter_retained_ranks(1).collect::<Vec<_>>(),
            vec![0],
        );
    }

    #[test]
    fn test_two_deposited() {
        let policy = StochasticPolicy::default();
        assert_eq!(policy.calc_num_strata_retained_exact(2), 2);
        assert_eq!(
            policy.iter_retained_ranks(2).collect::<Vec<_>>(),
            vec![0, 1],
        );
    }

    #[test]
    fn test_always_retains_first_and_last() {
        let policy = StochasticPolicy {
            retention_probability: 0.1,
        };
        for n in 1..50u64 {
            let retained: Vec<u64> = policy.iter_retained_ranks(n).collect();
            assert!(
                retained.contains(&0),
                "n={}: should retain rank 0, got {:?}",
                n,
                retained,
            );
            assert!(
                retained.contains(&(n - 1)),
                "n={}: should retain newest rank {}, got {:?}",
                n,
                n - 1,
                retained,
            );
        }
    }

    #[test]
    fn test_deterministic_reproducibility() {
        let policy = StochasticPolicy {
            retention_probability: 0.5,
        };
        let a: Vec<u64> = policy.iter_retained_ranks(30).collect();
        let b: Vec<u64> = policy.iter_retained_ranks(30).collect();
        assert_eq!(a, b, "should be deterministic/reproducible");
    }

    #[test]
    fn test_high_probability_retains_more() {
        let high = StochasticPolicy {
            retention_probability: 1.0,
        };
        let low = StochasticPolicy {
            retention_probability: 0.0,
        };
        let n = 50;
        let high_count = high.calc_num_strata_retained_exact(n);
        let low_count = low.calc_num_strata_retained_exact(n);
        assert!(
            high_count >= low_count,
            "higher probability should retain at least as many: {} vs {}",
            high_count,
            low_count,
        );
    }

    #[test]
    fn test_probability_one_retains_all() {
        let policy = StochasticPolicy {
            retention_probability: 1.0,
        };
        for n in 0..30u64 {
            assert_eq!(
                policy.calc_num_strata_retained_exact(n),
                n,
                "probability 1.0 should retain all at n={}",
                n,
            );
        }
    }

    #[test]
    fn test_gen_drop_ranks_only_drops_second_most_recent() {
        let policy = StochasticPolicy::default();
        for n in 3..30u64 {
            let retained: Vec<u64> = (0..n).collect();
            let drops = policy.gen_drop_ranks(n, &retained);
            assert!(
                drops.len() <= 1,
                "n={}: should drop at most 1, got {:?}",
                n,
                drops,
            );
            if !drops.is_empty() {
                assert_eq!(
                    drops[0],
                    n - 2,
                    "n={}: should only consider second-most-recent",
                    n,
                );
            }
        }
    }

    #[test]
    fn test_partial_eq_bitwise() {
        let a = StochasticPolicy {
            retention_probability: 0.5,
        };
        let b = StochasticPolicy {
            retention_probability: 0.5,
        };
        assert_eq!(a, b);

        let c = StochasticPolicy {
            retention_probability: f64::NAN,
        };
        let d = StochasticPolicy {
            retention_probability: f64::NAN,
        };
        // NaN != NaN normally, but bitwise they are equal (same bits).
        assert_eq!(c, d);

        // Positive and negative zero differ bitwise.
        let pos = StochasticPolicy {
            retention_probability: 0.0,
        };
        let neg = StochasticPolicy {
            retention_probability: -0.0,
        };
        assert_ne!(pos, neg);
    }

    #[test]
    fn test_algo_identifier() {
        let policy = StochasticPolicy::default();
        assert_eq!(policy.algo_identifier(), "stochastic_algo");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StochasticPolicy>();
    }

    #[test]
    fn test_retained_sorted_and_unique() {
        let policy = StochasticPolicy {
            retention_probability: 0.5,
        };
        for n in 0..50 {
            let retained: Vec<u64> = policy.iter_retained_ranks(n).collect();
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
    fn test_default() {
        let policy = StochasticPolicy::default();
        assert_eq!(policy.retention_probability, 0.5);
    }

    #[test]
    fn test_new_constructor() {
        let policy = StochasticPolicy::new(0.75);
        assert_eq!(policy.retention_probability, 0.75);
    }
}

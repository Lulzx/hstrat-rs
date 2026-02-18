use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;

/// Deterministic pseudo-stochastic retention policy.
///
/// Uses a hash-based deterministic rule to decide whether to retain the
/// second-most-recent stratum. The first (rank 0) and last (newest) strata
/// are always retained. All other strata that have already survived a
/// deposition step are never dropped.
///
/// The hash function is:
///   `(rank.wrapping_mul(hash_salt).wrapping_add(hash_salt)) % 2 == 0`
///
/// If the hash evaluates to true for the second-most-recent rank, that
/// stratum is retained; otherwise it is dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct PseudostochasticPolicy {
    pub hash_salt: u64,
}

impl PseudostochasticPolicy {
    pub fn new(hash_salt: u64) -> Self {
        Self { hash_salt }
    }

    /// Deterministic hash deciding whether to keep a rank.
    #[inline]
    fn should_retain(&self, rank: u64) -> bool {
        rank.wrapping_mul(self.hash_salt)
            .wrapping_add(self.hash_salt)
            .is_multiple_of(2)
    }
}

impl StratumRetentionPolicy for PseudostochasticPolicy {
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64> {
        if num_strata_deposited <= 2 {
            return Vec::new();
        }

        let newest_rank = num_strata_deposited - 1;
        let second_most_recent = newest_rank - 1;

        // Only consider dropping the second-most-recent stratum.
        // If it is rank 0, never drop it.
        if second_most_recent == 0 {
            return Vec::new();
        }

        // Check if the second-most-recent is actually in retained_ranks.
        if !retained_ranks.contains(&second_most_recent) {
            return Vec::new();
        }

        if self.should_retain(second_most_recent) {
            Vec::new()
        } else {
            alloc::vec![second_most_recent]
        }
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        if num_strata_deposited == 0 {
            return Box::new(core::iter::empty());
        }
        if num_strata_deposited == 1 {
            return Box::new(core::iter::once(0));
        }

        // Simulate the column from the beginning to determine which strata
        // were retained.
        let newest_rank = num_strata_deposited - 1;
        let mut retained = Vec::new();
        retained.push(0u64);

        for dep in 1..num_strata_deposited {
            retained.push(dep);

            if dep >= 2 {
                let second_most_recent = dep - 1;
                if second_most_recent != 0 && !self.should_retain(second_most_recent) {
                    retained.retain(|&r| r != second_most_recent);
                }
            }
        }

        // Ensure first and last are present.
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

    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64 {
        let ranks: Vec<u64> = self.iter_retained_ranks(num_strata_deposited).collect();
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        let retained: Vec<u64> = self.iter_retained_ranks(num_strata_deposited).collect();
        if retained.len() <= 1 {
            return num_strata_deposited.saturating_sub(1);
        }
        retained.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }

    fn algo_identifier(&self) -> &'static str {
        "pseudostochastic_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deposited() {
        let policy = PseudostochasticPolicy { hash_salt: 42 };
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        assert_eq!(policy.gen_drop_ranks(0, &[]), Vec::<u64>::new());
        assert_eq!(
            policy.iter_retained_ranks(0).collect::<Vec<_>>(),
            Vec::<u64>::new(),
        );
    }

    #[test]
    fn test_one_deposited() {
        let policy = PseudostochasticPolicy { hash_salt: 42 };
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        assert_eq!(policy.iter_retained_ranks(1).collect::<Vec<_>>(), vec![0],);
    }

    #[test]
    fn test_two_deposited() {
        let policy = PseudostochasticPolicy { hash_salt: 42 };
        assert_eq!(policy.calc_num_strata_retained_exact(2), 2);
        assert_eq!(
            policy.iter_retained_ranks(2).collect::<Vec<_>>(),
            vec![0, 1],
        );
    }

    #[test]
    fn test_always_retains_first_and_last() {
        let policy = PseudostochasticPolicy { hash_salt: 123 };
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
    fn test_deterministic() {
        let policy = PseudostochasticPolicy { hash_salt: 99 };
        let a: Vec<u64> = policy.iter_retained_ranks(20).collect();
        let b: Vec<u64> = policy.iter_retained_ranks(20).collect();
        assert_eq!(a, b, "should be deterministic");
    }

    #[test]
    fn test_different_salts_differ() {
        let p1 = PseudostochasticPolicy { hash_salt: 1 };
        let p2 = PseudostochasticPolicy { hash_salt: 2 };
        let mut found_diff = false;
        for n in 3..100 {
            let r1: Vec<u64> = p1.iter_retained_ranks(n).collect();
            let r2: Vec<u64> = p2.iter_retained_ranks(n).collect();
            if r1 != r2 {
                found_diff = true;
                break;
            }
        }
        assert!(
            found_diff,
            "different salts should produce different results",
        );
    }

    #[test]
    fn test_gen_drop_ranks_only_drops_second_most_recent() {
        let policy = PseudostochasticPolicy { hash_salt: 42 };
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
                    "n={}: should only drop second-most-recent",
                    n,
                );
            }
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = PseudostochasticPolicy { hash_salt: 0 };
        assert_eq!(policy.algo_identifier(), "pseudostochastic_algo");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PseudostochasticPolicy>();
    }

    #[test]
    fn test_retained_sorted_and_unique() {
        let policy = PseudostochasticPolicy { hash_salt: 77 };
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
    fn test_new_constructor() {
        let policy = PseudostochasticPolicy::new(42);
        assert_eq!(policy.hash_salt, 42);
    }
}

use alloc::vec::Vec;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::differentia::Differentia;
use crate::policies::StratumRetentionPolicy;

/// A single stratum: a rank paired with a random differentia fingerprint.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Stratum {
    pub rank: u64,
    pub differentia: Differentia,
}

/// A hereditary stratigraphic column.
///
/// Maintains an ordered sequence of strata (rank + differentia pairs).
/// When a new stratum is deposited the retention policy is consulted to
/// decide which older strata to prune, keeping memory bounded according
/// to the chosen policy.
#[derive(Clone, Debug)]
pub struct HereditaryStratigraphicColumn<P: StratumRetentionPolicy> {
    policy: P,
    differentia_bit_width: u8,
    strata: Vec<Stratum>,
    num_strata_deposited: u64,
    rng: SmallRng,
}

impl<P: StratumRetentionPolicy> HereditaryStratigraphicColumn<P> {
    /// Create a new column with a random seed.
    ///
    /// When the `std` feature is enabled, entropy is derived from the
    /// current system time.  In `no_std` builds, a fixed seed of `0` is
    /// used; prefer [`with_seed`](Self::with_seed) for deterministic
    /// reproducibility.
    pub fn new(policy: P, differentia_bit_width: u8) -> Self {
        debug_assert!(
            differentia_bit_width >= 1 && differentia_bit_width <= 64,
            "differentia_bit_width must be in 1..=64, got {}",
            differentia_bit_width
        );

        #[cfg(feature = "std")]
        let seed = {
            use std::time::SystemTime;
            let dur = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default();
            dur.as_nanos() as u64
        };

        #[cfg(not(feature = "std"))]
        let seed = 0u64;

        Self {
            policy,
            differentia_bit_width,
            strata: Vec::new(),
            num_strata_deposited: 0,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Create a new column with a deterministic seed for reproducibility.
    pub fn with_seed(policy: P, differentia_bit_width: u8, seed: u64) -> Self {
        debug_assert!(
            differentia_bit_width >= 1 && differentia_bit_width <= 64,
            "differentia_bit_width must be in 1..=64, got {}",
            differentia_bit_width
        );
        Self {
            policy,
            differentia_bit_width,
            strata: Vec::new(),
            num_strata_deposited: 0,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Reconstruct a column from its constituent parts (for deserialization).
    ///
    /// The caller is responsible for ensuring the strata are sorted by rank
    /// and consistent with the policy and num_strata_deposited.
    pub fn from_parts(
        policy: P,
        differentia_bit_width: u8,
        strata: Vec<Stratum>,
        num_strata_deposited: u64,
    ) -> Self {
        Self {
            policy,
            differentia_bit_width,
            strata,
            num_strata_deposited,
            rng: SmallRng::seed_from_u64(0), // RNG state is not preserved across serialization
        }
    }

    /// Deposit a single new stratum.
    ///
    /// Generates a random differentia, appends the stratum, increments the
    /// deposit counter, and then prunes strata that the policy says to drop.
    pub fn deposit_stratum(&mut self) {
        let mask = Differentia::mask(self.differentia_bit_width);
        let diff_value = self.rng.gen_range(0..=mask);
        let diff = Differentia::new(diff_value, self.differentia_bit_width);
        let rank = self.num_strata_deposited;
        self.strata.push(Stratum {
            rank,
            differentia: diff,
        });
        self.num_strata_deposited += 1;

        // Fast path: skip pruning if column already has the right count.
        let expected =
            self.policy
                .calc_num_strata_retained_exact(self.num_strata_deposited) as usize;
        if self.strata.len() <= expected {
            return;
        }

        // Common case: exactly one rank to drop (the former-newest that
        // is no longer aligned with the policy's spacing rule).
        // Remove it directly instead of scanning the entire column.
        if self.strata.len() == expected + 1 && self.strata.len() >= 2 {
            // The second-most-recent was the newest at the prior step
            // and is the most likely candidate for dropping.
            let idx = self.strata.len() - 2;
            self.strata.remove(idx);

            // Verify post-condition; if wrong, fall through to full scan
            if self.strata.len() == expected {
                return;
            }
        }

        // Slow path: bulk pruning (threshold crossings in some policies)
        let retained_ranks: Vec<u64> = self.strata.iter().map(|s| s.rank).collect();
        let drop_ranks =
            self.policy
                .gen_drop_ranks(self.num_strata_deposited, &retained_ranks);
        if !drop_ranks.is_empty() {
            let drop_set: alloc::collections::BTreeSet<u64> =
                drop_ranks.into_iter().collect();
            self.strata.retain(|s| !drop_set.contains(&s.rank));
        }
    }

    /// Deposit `n` strata in succession.
    pub fn deposit_strata(&mut self, n: u64) {
        for _ in 0..n {
            self.deposit_stratum();
        }
    }

    /// Clone this column and deposit one additional stratum on the clone,
    /// simulating a parent-child relationship.
    pub fn clone_descendant(&self) -> Self {
        let mut descendant = self.clone();
        descendant.deposit_stratum();
        descendant
    }

    /// Total number of strata that have been deposited (including pruned ones).
    pub fn get_num_strata_deposited(&self) -> u64 {
        self.num_strata_deposited
    }

    /// Number of strata currently retained in this column.
    pub fn get_num_strata_retained(&self) -> usize {
        self.strata.len()
    }

    /// The bit width of the differentia values in this column.
    pub fn get_stratum_differentia_bit_width(&self) -> u8 {
        self.differentia_bit_width
    }

    /// Reference to the retention policy used by this column.
    pub fn get_policy(&self) -> &P {
        &self.policy
    }

    /// Iterate over the ranks of all retained strata, in ascending order.
    pub fn iter_retained_ranks(&self) -> impl Iterator<Item = u64> + '_ {
        self.strata.iter().map(|s| s.rank)
    }

    /// Iterate over the differentia values of all retained strata, in rank order.
    pub fn iter_retained_differentia(&self) -> impl Iterator<Item = Differentia> + '_ {
        self.strata.iter().map(|s| s.differentia)
    }

    /// Iterate over references to all retained strata.
    pub fn iter_retained_strata(&self) -> impl Iterator<Item = &Stratum> {
        self.strata.iter()
    }

    /// Get the stratum at a given column index (0-based position in the
    /// retained strata vector).
    pub fn get_stratum_at_column_index(&self, index: usize) -> Option<&Stratum> {
        self.strata.get(index)
    }

    /// Get the stratum at a given rank via binary search.
    ///
    /// Returns `None` if no stratum with the given rank is currently retained.
    pub fn get_stratum_at_rank(&self, rank: u64) -> Option<&Stratum> {
        self.strata
            .binary_search_by_key(&rank, |s| s.rank)
            .ok()
            .map(|idx| &self.strata[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::{FixedResolutionPolicy, PerfectResolutionPolicy};

    #[test]
    fn test_perfect_resolution_retains_all() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        col.deposit_strata(100);
        assert_eq!(col.get_num_strata_deposited(), 100);
        assert_eq!(col.get_num_strata_retained(), 100);

        // All ranks 0..100 should be present
        let ranks: Vec<u64> = col.iter_retained_ranks().collect();
        let expected: Vec<u64> = (0..100).collect();
        assert_eq!(ranks, expected);
    }

    #[test]
    fn test_fixed_resolution_correct_retention() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(10),
            64,
            42,
        );
        col.deposit_strata(100);
        assert_eq!(col.get_num_strata_deposited(), 100);

        // With resolution=10, retained ranks should be 0, 10, 20, ..., 90, 99
        let ranks: Vec<u64> = col.iter_retained_ranks().collect();
        // Check that 0 and 99 are present
        assert_eq!(*ranks.first().unwrap(), 0);
        assert_eq!(*ranks.last().unwrap(), 99);
        // Check that ranks are multiples of 10 plus the newest rank (99)
        for &r in &ranks {
            assert!(
                r % 10 == 0 || r == 99,
                "unexpected rank {} retained",
                r
            );
        }
        // Expected: 0,10,20,30,40,50,60,70,80,90,99 = 11
        assert_eq!(col.get_num_strata_retained(), 11);
    }

    #[test]
    fn test_clone_descendant() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        col.deposit_strata(10);
        assert_eq!(col.get_num_strata_deposited(), 10);

        let descendant = col.clone_descendant();
        assert_eq!(descendant.get_num_strata_deposited(), 11);
        assert_eq!(descendant.get_num_strata_retained(), 11);

        // The descendant should have all the parent's strata plus one more
        let parent_ranks: Vec<u64> = col.iter_retained_ranks().collect();
        let desc_ranks: Vec<u64> = descendant.iter_retained_ranks().collect();
        for &r in &parent_ranks {
            assert!(
                desc_ranks.contains(&r),
                "descendant missing parent rank {}",
                r
            );
        }
        assert_eq!(*desc_ranks.last().unwrap(), 10);
    }

    #[test]
    fn test_get_stratum_at_rank_binary_search() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        col.deposit_strata(50);

        // Every rank 0..50 should be findable
        for rank in 0..50 {
            let stratum = col.get_stratum_at_rank(rank);
            assert!(stratum.is_some(), "missing stratum at rank {}", rank);
            assert_eq!(stratum.unwrap().rank, rank);
        }

        // Non-existent rank
        assert!(col.get_stratum_at_rank(50).is_none());
        assert!(col.get_stratum_at_rank(999).is_none());
    }

    #[test]
    fn test_get_stratum_at_rank_with_pruning() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(10),
            64,
            42,
        );
        col.deposit_strata(100);

        // Rank 0 should be retained
        assert!(col.get_stratum_at_rank(0).is_some());
        // Rank 10 should be retained
        assert!(col.get_stratum_at_rank(10).is_some());
        // Rank 99 (newest) should be retained
        assert!(col.get_stratum_at_rank(99).is_some());
        // Rank 5 should have been pruned
        assert!(col.get_stratum_at_rank(5).is_none());
    }

    #[test]
    fn test_zero_deposits() {
        let col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        assert_eq!(col.get_num_strata_deposited(), 0);
        assert_eq!(col.get_num_strata_retained(), 0);
        assert!(col.iter_retained_ranks().next().is_none());
        assert!(col.iter_retained_differentia().next().is_none());
        assert!(col.iter_retained_strata().next().is_none());
        assert!(col.get_stratum_at_column_index(0).is_none());
        assert!(col.get_stratum_at_rank(0).is_none());
    }

    #[test]
    fn test_bit_width_1() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 1, 123);
        col.deposit_strata(20);
        assert_eq!(col.get_num_strata_deposited(), 20);
        assert_eq!(col.get_num_strata_retained(), 20);
        assert_eq!(col.get_stratum_differentia_bit_width(), 1);

        // All differentia values should be 0 or 1
        for d in col.iter_retained_differentia() {
            assert!(
                d.value() == 0 || d.value() == 1,
                "bit_width=1 differentia should be 0 or 1, got {}",
                d.value()
            );
        }
    }

    #[test]
    fn test_bit_width_64() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 7);
        col.deposit_strata(10);
        assert_eq!(col.get_num_strata_deposited(), 10);
        assert_eq!(col.get_num_strata_retained(), 10);
        assert_eq!(col.get_stratum_differentia_bit_width(), 64);
    }

    #[test]
    fn test_get_policy() {
        let policy = FixedResolutionPolicy::new(7);
        let col = HereditaryStratigraphicColumn::with_seed(policy.clone(), 64, 1);
        assert_eq!(*col.get_policy(), policy);
    }

    #[test]
    fn test_iter_retained_strata_matches_ranks_and_differentia() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 32, 55);
        col.deposit_strata(30);

        let ranks: Vec<u64> = col.iter_retained_ranks().collect();
        let diffs: Vec<Differentia> = col.iter_retained_differentia().collect();
        let strata: Vec<&Stratum> = col.iter_retained_strata().collect();

        assert_eq!(strata.len(), ranks.len());
        assert_eq!(strata.len(), diffs.len());

        for (i, s) in strata.iter().enumerate() {
            assert_eq!(s.rank, ranks[i]);
            assert_eq!(s.differentia, diffs[i]);
        }
    }

    #[test]
    fn test_get_stratum_at_column_index() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        col.deposit_strata(10);

        for i in 0..10 {
            let s = col.get_stratum_at_column_index(i).unwrap();
            assert_eq!(s.rank, i as u64);
        }
        assert!(col.get_stratum_at_column_index(10).is_none());
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let mut col1 =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        let mut col2 =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        col1.deposit_strata(50);
        col2.deposit_strata(50);

        let d1: Vec<u64> = col1.iter_retained_differentia().map(|d| d.value()).collect();
        let d2: Vec<u64> = col2.iter_retained_differentia().map(|d| d.value()).collect();
        assert_eq!(d1, d2, "same seed should produce identical differentia");
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut col1 =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 1);
        let mut col2 =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 2);
        col1.deposit_strata(50);
        col2.deposit_strata(50);

        let d1: Vec<u64> = col1.iter_retained_differentia().map(|d| d.value()).collect();
        let d2: Vec<u64> = col2.iter_retained_differentia().map(|d| d.value()).collect();
        // With 64-bit differentia, the chance of all 50 being equal with different seeds is 0
        assert_ne!(d1, d2, "different seeds should produce different differentia");
    }
}

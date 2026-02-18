use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use super::sorted_set_difference;
use crate::bit_floor;

/// Retains strata using a geometric sequence based on the nth-root of the
/// number of depositions.
///
/// The `degree` parameter controls the exponent, and `interspersal` controls
/// how many strata are retained between each geometric target point.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricSeqNthRootPolicy {
    pub degree: u64,
    pub interspersal: u64,
}

impl GeometricSeqNthRootPolicy {
    pub fn new(degree: u64, interspersal: u64) -> Self {
        assert!(degree > 0, "degree must be positive");
        assert!(interspersal > 0, "interspersal must be positive");
        Self {
            degree,
            interspersal,
        }
    }

    /// Create with default interspersal of 2.
    pub fn with_degree(degree: u64) -> Self {
        Self::new(degree, 2)
    }
}

/// Compute x^y using the platform math library when available (matching
/// Python's behavior), falling back to libm for no_std.
#[inline]
fn powf_compat(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::pow(x, y)
    }
}

/// Calculate the common ratio: `num_strata_deposited^(1/degree)`.
/// Matches Python hstrat's `_calc_common_ratio`.
fn calc_common_ratio(degree: u64, num_strata_deposited: u64) -> f64 {
    powf_compat(num_strata_deposited as f64, 1.0 / degree as f64)
}

/// Compute the set of retained ranks for this policy.
///
/// Matches Python hstrat's `_get_retained_ranks` for `geom_seq_nth_root_algo`:
/// For each power 1..=degree, computes a target recency, rank cutoff,
/// backstop, and separation, then retains `range(backstop, n, sep)`.
fn compute_retained_ranks(degree: u64, interspersal: u64, num_strata_deposited: u64) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }

    let last_rank = num_strata_deposited - 1;
    let mut retained = alloc::collections::BTreeSet::new();
    retained.insert(0u64);
    retained.insert(last_rank);

    let common_ratio = calc_common_ratio(degree, num_strata_deposited);

    for pow in 1..=degree {
        // Compute common_ratio^pow using repeated multiplication instead of
        // libm::pow to match Python's behavior. Python's ** operator for
        // float**int uses repeated multiplication, which can be more precise
        // than the log-exp approach of C's pow(). For example,
        // `16^(1/3) * 16^(1/3) * 16^(1/3) = 16.0` exactly, while
        // `pow(16^(1/3), 3.0) = 15.9999...` due to intermediate rounding.
        let mut target_recency = 1.0f64;
        for _ in 0..pow {
            target_recency *= common_ratio;
        }

        // target_rank (Python: iter_target_ranks)
        let recency_ceil = libm::ceil(target_recency) as u64;
        let _target_rank = num_strata_deposited.saturating_sub(recency_ceil);

        // rank_sep (Python: iter_rank_seps)
        // Python: bit_floor(int(max(target_recency / interspersal, 1.0)))
        let target_sep = if target_recency / interspersal as f64 > 1.0 {
            target_recency / interspersal as f64
        } else {
            1.0
        };
        let retained_ranks_sep = bit_floor(target_sep as u64).max(1);

        // rank_cutoff (Python: iter_rank_cutoffs)
        let extended_recency = target_recency * (interspersal + 1) as f64 / interspersal as f64;
        let extended_ceil = libm::ceil(extended_recency) as u64;
        let rank_cutoff = num_strata_deposited.saturating_sub(extended_ceil);

        // backstop: round UP rank_cutoff to multiple of retained_ranks_sep
        // Python: rank_cutoff - (rank_cutoff % -retained_ranks_sep)
        // which is ceiling division to next multiple
        let min_retained_rank = if rank_cutoff.is_multiple_of(retained_ranks_sep) {
            rank_cutoff
        } else {
            rank_cutoff + retained_ranks_sep - (rank_cutoff % retained_ranks_sep)
        };

        // Retain range(min_retained_rank, num_strata_deposited, retained_ranks_sep)
        let mut r = min_retained_rank;
        while r < num_strata_deposited {
            retained.insert(r);
            r += retained_ranks_sep;
        }
    }

    retained.into_iter().collect()
}

impl StratumRetentionPolicy for GeometricSeqNthRootPolicy {
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let keep = compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited);
        sorted_set_difference(retained_ranks, &keep)
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        let ranks = compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited);
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited).len() as u64
    }

    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64 {
        let ranks = compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited);
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        let ranks = compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited);
        if ranks.len() <= 1 {
            return 0;
        }
        ranks.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }

    fn algo_identifier(&self) -> &'static str {
        "geom_seq_nth_root_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_small_depositions() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        // 4 depositions (degree=3, so we retain first 3 + newest)
        let ranks: Vec<u64> = policy.iter_retained_ranks(4).collect();
        assert!(ranks.contains(&0));
        assert!(ranks.contains(&3)); // newest
    }

    #[test]
    fn test_endpoints_always_retained() {
        let policy = GeometricSeqNthRootPolicy::new(4, 2);
        for n in 1..100 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            assert_eq!(*ranks.first().unwrap(), 0, "rank 0 missing at n={n}");
            assert_eq!(
                *ranks.last().unwrap(),
                n - 1,
                "newest rank missing at n={n}"
            );
        }
    }

    #[test]
    fn test_retained_count_consistency() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        for n in 0..100 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_ranks_are_sorted() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        for n in 0..200 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            for w in ranks.windows(2) {
                assert!(w[0] < w[1], "not sorted at n={n}: {:?}", ranks);
            }
        }
    }

    #[test]
    fn test_with_degree_constructor() {
        let p = GeometricSeqNthRootPolicy::with_degree(5);
        assert_eq!(p.degree, 5);
        assert_eq!(p.interspersal, 2);
    }

    #[test]
    fn test_gen_drop_ranks_correct_set() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        let retained = compute_retained_ranks(3, 2, 50);
        let dropped = policy.gen_drop_ranks(50, &retained);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(50).collect();
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(policy.calc_rank_at_column_index(i, 50), r);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        assert_eq!(policy.algo_identifier(), "geom_seq_nth_root_algo");
    }

    #[test]
    fn test_rank_0_always_retained() {
        let policy = GeometricSeqNthRootPolicy::new(5, 2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(100).collect();
        assert!(ranks.contains(&0), "rank 0 should be retained");
    }

    #[test]
    fn test_large_depositions() {
        let policy = GeometricSeqNthRootPolicy::new(3, 2);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1000).collect();
        assert!(!ranks.is_empty());
        assert_eq!(*ranks.first().unwrap(), 0);
        assert_eq!(*ranks.last().unwrap(), 999);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeometricSeqNthRootPolicy>();
    }
}

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::geometric_seq_nth_root::GeometricSeqNthRootPolicy;
use super::r#trait::StratumRetentionPolicy;
use super::recency_proportional::RecencyProportionalPolicy;

/// Retains strata using a recency-proportional approach with an upper bound
/// on the number of retained strata.
///
/// Dynamically picks between `RecencyProportionalPolicy` and
/// `GeometricSeqNthRootPolicy` depending on the number of depositions and
/// the `size_curb` budget.
///
/// Matches Python hstrat's `recency_proportional_resolution_curbed_algo`.
#[derive(Clone, Debug, PartialEq)]
pub struct CurbedRecencyProportionalPolicy {
    pub size_curb: u64,
}

impl CurbedRecencyProportionalPolicy {
    pub fn new(size_curb: u64) -> Self {
        assert!(size_curb >= 8, "size_curb must be at least 8");
        Self { size_curb }
    }
}

/// Compute the provided resolution for a given size_curb and number of
/// depositions.
///
/// Python: `size_curb // (int(n).bit_length() + 1) - 1`
/// Returns -1 (as `None`) if below threshold (2).
fn calc_provided_resolution(size_curb: u64, num_strata_deposited: u64) -> Option<u64> {
    let bit_len = 64 - num_strata_deposited.leading_zeros() as u64;
    let res = size_curb / (bit_len + 1);
    if res < 1 {
        return None;
    }
    let resolution = res - 1;
    // Threshold: resolution must be >= 2
    if resolution >= 2 {
        Some(resolution)
    } else {
        None
    }
}

/// Compute the provided degree for the geom_seq_nth_root fallback.
///
/// Python: `max((size_curb - 2) // (2 * interspersal + 2), 1)`
fn calc_provided_degree(size_curb: u64, interspersal: u64) -> u64 {
    ((size_curb - 2) / (2 * interspersal + 2)).max(1)
}

/// Compute the set of retained ranks for the curbed policy by delegating to
/// either RecencyProportionalPolicy or GeometricSeqNthRootPolicy.
fn compute_retained_ranks(size_curb: u64, num_strata_deposited: u64) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }

    match calc_provided_resolution(size_curb, num_strata_deposited) {
        Some(resolution) => {
            // Use recency-proportional with computed resolution
            let policy = RecencyProportionalPolicy::new(resolution);
            policy.iter_retained_ranks(num_strata_deposited).collect()
        }
        None => {
            // Use geom_seq_nth_root with computed degree and interspersal=2
            let interspersal = 2u64;
            let degree = calc_provided_degree(size_curb, interspersal);
            let policy = GeometricSeqNthRootPolicy::new(degree, interspersal);
            policy.iter_retained_ranks(num_strata_deposited).collect()
        }
    }
}

impl StratumRetentionPolicy for CurbedRecencyProportionalPolicy {
    fn gen_drop_ranks(&self, num_strata_deposited: u64, retained_ranks: &[u64]) -> Vec<u64> {
        if num_strata_deposited == 0 {
            return Vec::new();
        }
        let keep = compute_retained_ranks(self.size_curb, num_strata_deposited);
        retained_ranks
            .iter()
            .copied()
            .filter(|rank| !keep.contains(rank))
            .collect()
    }

    fn iter_retained_ranks(&self, num_strata_deposited: u64) -> Box<dyn Iterator<Item = u64> + '_> {
        let ranks = compute_retained_ranks(self.size_curb, num_strata_deposited);
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(&self, num_strata_deposited: u64) -> u64 {
        compute_retained_ranks(self.size_curb, num_strata_deposited).len() as u64
    }

    fn calc_rank_at_column_index(&self, index: usize, num_strata_deposited: u64) -> u64 {
        let ranks = compute_retained_ranks(self.size_curb, num_strata_deposited);
        ranks[index]
    }

    fn calc_mrca_uncertainty_abs_exact(&self, num_strata_deposited: u64) -> u64 {
        if num_strata_deposited <= 1 {
            return 0;
        }
        let ranks = compute_retained_ranks(self.size_curb, num_strata_deposited);
        if ranks.len() <= 1 {
            return 0;
        }
        ranks.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }

    fn algo_identifier(&self) -> &'static str {
        "recency_proportional_resolution_curbed_algo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depositions() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        assert_eq!(policy.calc_num_strata_retained_exact(0), 0);
        let ranks: Vec<u64> = policy.iter_retained_ranks(0).collect();
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_one_deposition() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        assert_eq!(policy.calc_num_strata_retained_exact(1), 1);
        let ranks: Vec<u64> = policy.iter_retained_ranks(1).collect();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_endpoints_always_retained() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
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
        let policy = CurbedRecencyProportionalPolicy::new(16);
        for n in 0..150 {
            let count = policy.calc_num_strata_retained_exact(n);
            let iter_count = policy.iter_retained_ranks(n).count() as u64;
            assert_eq!(count, iter_count, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_ranks_are_sorted() {
        let policy = CurbedRecencyProportionalPolicy::new(12);
        for n in 0..200 {
            let ranks: Vec<u64> = policy.iter_retained_ranks(n).collect();
            for w in ranks.windows(2) {
                assert!(w[0] < w[1], "not sorted at n={n}: {:?}", ranks);
            }
        }
    }

    #[test]
    fn test_gen_drop_ranks_correct_set() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        let retained = compute_retained_ranks(10, 50);
        let dropped = policy.gen_drop_ranks(50, &retained);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_calc_rank_at_column_index() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        let ranks: Vec<u64> = policy.iter_retained_ranks(50).collect();
        for (i, &r) in ranks.iter().enumerate() {
            assert_eq!(policy.calc_rank_at_column_index(i, 50), r);
        }
    }

    #[test]
    fn test_algo_identifier() {
        let policy = CurbedRecencyProportionalPolicy::new(10);
        assert_eq!(
            policy.algo_identifier(),
            "recency_proportional_resolution_curbed_algo"
        );
    }

    #[test]
    #[should_panic(expected = "size_curb must be at least 8")]
    fn test_size_curb_minimum() {
        CurbedRecencyProportionalPolicy::new(5);
    }
}

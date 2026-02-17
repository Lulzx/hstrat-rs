use alloc::boxed::Box;
use alloc::vec::Vec;

use super::r#trait::StratumRetentionPolicy;
use crate::bit_floor;

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
}

// ---------------------------------------------------------------------------
// Floating-point helpers (matching the non-tapered variant)
// ---------------------------------------------------------------------------

/// Compute x^y using platform pow when available (matching Python), else libm.
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

/// `num_strata_deposited^(1/degree)` -- Python's `_calc_common_ratio`.
fn calc_common_ratio(degree: u64, num_strata_deposited: u64) -> f64 {
    powf_compat(num_strata_deposited as f64, 1.0 / degree as f64)
}

/// `common_ratio^pow` via repeated multiplication (matches Python `**` for
/// float**int).
fn calc_target_recency(degree: u64, pow: u64, num_strata_deposited: u64) -> f64 {
    let cr = calc_common_ratio(degree, num_strata_deposited);
    let mut res = 1.0f64;
    for _ in 0..pow {
        res *= cr;
    }
    res
}

/// `bit_floor(int(max(recency / interspersal, 1.0)))` -- Python's
/// `_calc_rank_sep`.
fn calc_rank_sep(
    degree: u64,
    interspersal: u64,
    pow: u64,
    num_strata_deposited: u64,
) -> u64 {
    let target_recency =
        calc_target_recency(degree, pow, num_strata_deposited);
    let target_sep = if target_recency / interspersal as f64 > 1.0 {
        target_recency / interspersal as f64
    } else {
        1.0
    };
    bit_floor(target_sep as u64).max(1)
}

/// Python's `_calc_rank_cutoff`.
fn calc_rank_cutoff(
    degree: u64,
    interspersal: u64,
    pow: u64,
    num_strata_deposited: u64,
) -> u64 {
    let target_recency =
        calc_target_recency(degree, pow, num_strata_deposited);
    let extended =
        target_recency * (interspersal + 1) as f64 / interspersal as f64;
    let extended_ceil = libm::ceil(extended) as u64;
    if extended_ceil >= num_strata_deposited {
        0
    } else {
        num_strata_deposited - extended_ceil
    }
}

/// Python's `_calc_rank_backstop` -- round UP `rank_cutoff` to a multiple of
/// `rank_sep`.
fn calc_rank_backstop(
    degree: u64,
    interspersal: u64,
    pow: u64,
    num_strata_deposited: u64,
) -> u64 {
    let cutoff = calc_rank_cutoff(degree, interspersal, pow, num_strata_deposited);
    let sep = calc_rank_sep(degree, interspersal, pow, num_strata_deposited);
    // Round UP to next multiple of sep (Python: `cutoff - (cutoff % -sep)`)
    if cutoff % sep == 0 {
        cutoff
    } else {
        cutoff + sep - (cutoff % sep)
    }
}

// ---------------------------------------------------------------------------
// Search helpers (porting Python `interval_search`)
// ---------------------------------------------------------------------------

/// Binary search: find the smallest `x` in `[lo, hi]` where `pred(x)` is true.
/// Returns `None` if no such x exists.
fn binary_search_first<F: Fn(u64) -> bool>(
    pred: &F,
    lo: u64,
    hi: u64,
) -> Option<u64> {
    let mut lo = lo;
    let mut hi = hi;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo > hi {
        return None;
    }
    if pred(lo) {
        Some(lo)
    } else {
        None
    }
}

/// Doubling search: find the smallest `x >= lower_bound` where `pred(x)` is
/// true. Doubles `bound` until `pred(lower_bound + bound)` holds, then binary
/// searches.
fn doubling_search<F: Fn(u64) -> bool>(pred: &F, lower_bound: u64) -> u64 {
    let mut bound: u64 = 1;
    while !pred(lower_bound.saturating_add(bound)) {
        bound = bound.saturating_mul(2);
        // Safety valve for extremely large searches
        if bound > 1u64 << 50 {
            break;
        }
    }
    let prev_bound = bound / 2;
    let lo = lower_bound.saturating_add(prev_bound);
    let hi = lower_bound.saturating_add(bound);
    binary_search_first(pred, lo, hi).unwrap_or(hi)
}

/// `div_range(start, end, divide_by)` yields `start, start/d, start/d^2, ...`
/// while value > end.
fn div_range(start: u64, end: u64, divide_by: u64) -> Vec<u64> {
    let mut result = Vec::new();
    let mut val = start;
    while val > end {
        result.push(val);
        val /= divide_by;
    }
    result
}

// ---------------------------------------------------------------------------
// Priority rank generation (porting Python `_iter_priority_ranks`)
// ---------------------------------------------------------------------------

/// Generate the priority-ordered sequence of ranks for a given `pow` at
/// `num_strata_deposited`. Returns ranks from highest priority
/// (last-to-be-deleted) to lowest priority (first-to-be-deleted).
///
/// This is a faithful port of Python hstrat's `_iter_priority_ranks`.
fn iter_priority_ranks(
    degree: u64,
    interspersal: u64,
    pow: u64,
    num_strata_deposited: u64,
) -> Vec<u64> {
    let d = degree;
    let i = interspersal;
    let n = num_strata_deposited;

    if n == 0 {
        return Vec::new();
    }

    let min_retained_rank = if pow > 0 {
        calc_rank_backstop(d, i, pow, n)
    } else {
        0
    };
    let retained_ranks_sep = calc_rank_sep(d, i, pow, n);

    // pow == 0: just reversed(range(n))
    if pow == 0 {
        return (0..n).rev().collect();
    }

    let mut result = Vec::new();

    let biggest_relevant_rank;
    let biggest_relevant_sep;

    if pow == degree {
        // Special case for highest degree: use calc_rank_sep threshold
        biggest_relevant_rank = doubling_search(
            &|x: u64| calc_rank_sep(d, i, pow, x + 1) >= n,
            n,
        );
        biggest_relevant_sep =
            calc_rank_sep(d, i, pow, biggest_relevant_rank);
    } else {
        biggest_relevant_rank = doubling_search(
            &|x: u64| calc_rank_backstop(d, i, pow, x + 1) >= n,
            n,
        );
        biggest_relevant_sep =
            calc_rank_sep(d, i, pow, biggest_relevant_rank);
    }

    // For each cur_sep in div_range(biggest_relevant_sep, retained_ranks_sep, 2)
    for cur_sep in div_range(biggest_relevant_sep, retained_ranks_sep, 2) {
        let cur_sep_rank = doubling_search(
            &|x: u64| calc_rank_sep(d, i, pow, x) >= cur_sep,
            n.max(cur_sep),
        );
        let cur_sep_rank_backstop =
            calc_rank_backstop(d, i, pow, cur_sep_rank);

        // yield from reversed(range(backstop, min(cur_sep_rank+1, n), cur_sep))
        let stop = (cur_sep_rank + 1).min(n);
        if cur_sep > 0 && cur_sep_rank_backstop < stop {
            let mut r = cur_sep_rank_backstop;
            let mut forward = Vec::new();
            while r < stop {
                forward.push(r);
                r += cur_sep;
            }
            result.extend(forward.into_iter().rev());
        }
    }

    // yield from reversed(range(min_retained_rank, n, retained_ranks_sep))
    {
        let mut r = min_retained_rank;
        let mut forward = Vec::new();
        while r < n {
            forward.push(r);
            r += retained_ranks_sep;
        }
        result.extend(forward.into_iter().rev());
    }

    // Recurse
    if retained_ranks_sep == 1 {
        // base case: yield from reversed(range(0, min_retained_rank))
        result.extend((0..min_retained_rank).rev());
        return result;
    }

    // Python: prev_sep_rank = binary_search(
    //   lambda x: calc_rank_sep(d, i, pow, x+1) >= retained_ranks_sep,
    //   0, n-1)
    let prev_sep_rank = binary_search_first(
        &|x: u64| calc_rank_sep(d, i, pow, x + 1) >= retained_ranks_sep,
        0,
        n - 1,
    )
    .unwrap_or(n - 1);

    // Python: yield from range(min_retained_rank, prev_sep_rank, -retained_ranks_sep)
    // range(start, stop, negative_step) in Python goes from start downward
    // BUT this Python range is (min_retained_rank, prev_sep_rank, -retained_ranks_sep)
    // Since min_retained_rank < prev_sep_rank typically, and step is negative,
    // this range is empty! (Python range with negative step requires start > stop)
    // So this typically yields nothing.
    if min_retained_rank > prev_sep_rank {
        let mut r = min_retained_rank;
        while r > prev_sep_rank {
            result.push(r);
            r = r.saturating_sub(retained_ranks_sep);
            if r == 0 && min_retained_rank > 0 {
                break;
            }
        }
    }

    // Recurse: iter_priority_ranks(d, i, pow, min(prev_sep_rank+1, n-1))
    let recurse_n = (prev_sep_rank + 1).min(n - 1);
    let sub = iter_priority_ranks(d, i, pow, recurse_n);
    result.extend(sub);

    result
}

// ---------------------------------------------------------------------------
// Main retained-ranks computation
// ---------------------------------------------------------------------------

/// Upper bound on retained strata count.
/// Python: `degree * 2 * (interspersal + 1) + 2`, capped at n.
fn calc_upper_bound(degree: u64, interspersal: u64, num_strata_deposited: u64) -> u64 {
    let ub = degree * 2 * (interspersal + 1) + 2;
    ub.min(num_strata_deposited)
}

/// Compute the set of retained ranks for the tapered policy.
///
/// Faithfully ports Python hstrat's `_get_retained_ranks` for
/// `geom_seq_nth_root_tapered_algo`.
fn compute_retained_ranks(
    degree: u64,
    interspersal: u64,
    num_strata_deposited: u64,
) -> Vec<u64> {
    if num_strata_deposited == 0 {
        return Vec::new();
    }

    let n = num_strata_deposited;
    let upper_bound = calc_upper_bound(degree, interspersal, n);

    // If n <= upper_bound, retain everything
    if n <= upper_bound {
        return (0..n).collect();
    }

    let last_rank = n - 1;
    let mut res = alloc::collections::BTreeSet::new();
    res.insert(0u64);
    res.insert(last_rank);

    // Pre-generate priority rank sequences for pow = degree down to 1
    let mut priority_seqs: Vec<Vec<u64>> = Vec::new();
    for pow in (1..=degree).rev() {
        priority_seqs.push(iter_priority_ranks(
            degree,
            interspersal,
            pow,
            n,
        ));
    }
    // Track current position in each sequence
    let mut positions: Vec<usize> = alloc::vec![0usize; priority_seqs.len()];

    // Round-robin: draw one new rank from each iterator per round
    loop {
        if res.len() as u64 >= upper_bound {
            break;
        }
        let res_before = res.len();

        for (idx, seq) in priority_seqs.iter().enumerate() {
            // Draw from this iterator until we find a rank not already in res
            while positions[idx] < seq.len() {
                let rank = seq[positions[idx]];
                positions[idx] += 1;
                if !res.contains(&rank) {
                    res.insert(rank);
                    break;
                }
            }
            // Check upper bound after each iterator
            if res.len() as u64 >= upper_bound {
                break;
            }
        }

        // If no progress, all iterators are exhausted
        if res.len() == res_before {
            break;
        }
    }

    // Fill remaining from pow=0 (reversed(range(n)))
    if res.len() < upper_bound as usize {
        let pow0_seq = iter_priority_ranks(degree, interspersal, 0, n);
        for rank in pow0_seq {
            if res.len() as u64 >= upper_bound {
                break;
            }
            res.insert(rank);
        }
    }

    res.into_iter().collect()
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
            compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited);

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
        let ranks = compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        );
        Box::new(ranks.into_iter())
    }

    fn calc_num_strata_retained_exact(
        &self,
        num_strata_deposited: u64,
    ) -> u64 {
        compute_retained_ranks(
            self.degree,
            self.interspersal,
            num_strata_deposited,
        )
        .len() as u64
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
            compute_retained_ranks(self.degree, self.interspersal, num_strata_deposited);
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

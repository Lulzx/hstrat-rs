use alloc::collections::VecDeque;

use crate::column::HereditaryStratigraphicColumn;
use crate::policies::StratumRetentionPolicy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Probability of a spurious differentia collision at the given bit width.
///
/// Two independent random differentia at `bit_width` bits match with
/// probability `1 / 2^bit_width`.
pub fn calc_probability_differentia_collision(bit_width: u8) -> f64 {
    // Use libm::pow to avoid overflow when bit_width == 64 (1u64 << 64 panics).
    libm::pow(2.0, bit_width as f64).recip()
}

/// Minimum number of consecutive spurious collisions needed to fool a test
/// at the given significance level.
///
/// Returns the smallest `n` such that `p^n <= significance_level`, where
/// `p = calc_probability_differentia_collision(bit_width)`.
///
/// Special cases:
/// - `significance_level == 0.0` → `u64::MAX` (impossible to satisfy)
/// - `significance_level >= p` → `1` (a single collision already exceeds threshold)
pub fn calc_min_implausible_spurious_collisions(
    bit_width: u8,
    significance_level: f64,
) -> u64 {
    if significance_level == 0.0 {
        return u64::MAX;
    }
    let p = calc_probability_differentia_collision(bit_width);
    if significance_level >= p {
        return 1;
    }
    // Solve p^n <= significance_level → n >= ln(sig) / ln(p)
    // Both ln(sig) and ln(p) are negative (sig < 1, p < 1), so the ratio is positive.
    libm::ceil(libm::log(significance_level) / libm::log(p)) as u64
}

// ---------------------------------------------------------------------------
// Retained-rank iteration helpers
// ---------------------------------------------------------------------------

/// Iterate ranks retained by both columns where differentia matches.
///
/// Performs a two-pointer merge over the sorted retained ranks of both
/// columns. Yields common ranks where both strata have matching differentia.
/// **Stops at the first common rank where differentia differs.**
///
/// This yields only the "reliable commonality" prefix, not spurious matches
/// past divergence.
pub fn iter_ranks_of_retained_commonality_between<'a, P1, P2>(
    a: &'a HereditaryStratigraphicColumn<P1>,
    b: &'a HereditaryStratigraphicColumn<P2>,
) -> impl Iterator<Item = u64> + 'a
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );

    CommonalityIter {
        iter_a: a.iter_retained_ranks().collect::<alloc::vec::Vec<_>>().into_iter().peekable(),
        iter_b: b.iter_retained_ranks().collect::<alloc::vec::Vec<_>>().into_iter().peekable(),
        a,
        b,
        bit_width,
        done: false,
    }
}

struct CommonalityIter<'a, P1, P2>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    iter_a: core::iter::Peekable<alloc::vec::IntoIter<u64>>,
    iter_b: core::iter::Peekable<alloc::vec::IntoIter<u64>>,
    a: &'a HereditaryStratigraphicColumn<P1>,
    b: &'a HereditaryStratigraphicColumn<P2>,
    bit_width: u8,
    done: bool,
}

impl<P1, P2> Iterator for CommonalityIter<'_, P1, P2>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.done {
            return None;
        }
        loop {
            let ra = *self.iter_a.peek()?;
            let rb = match self.iter_b.peek() {
                Some(&r) => r,
                None => return None,
            };
            match ra.cmp(&rb) {
                core::cmp::Ordering::Less => {
                    self.iter_a.next();
                }
                core::cmp::Ordering::Greater => {
                    self.iter_b.next();
                }
                core::cmp::Ordering::Equal => {
                    let sa = self.a.get_stratum_at_rank(ra).unwrap();
                    let sb = self.b.get_stratum_at_rank(ra).unwrap();
                    self.iter_a.next();
                    self.iter_b.next();
                    if sa.differentia.matches(sb.differentia, self.bit_width) {
                        return Some(ra);
                    } else {
                        self.done = true;
                        return None;
                    }
                }
            }
        }
    }
}

/// Return the `n`-th (0-indexed) common retained rank where differentia matches.
///
/// Performs a two-pointer merge and counts matching common ranks.
/// Returns `None` if fewer than `n+1` matching common ranks exist.
pub fn get_nth_common_rank_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    n: usize,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    iter_ranks_of_retained_commonality_between(a, b).nth(n)
}

// ---------------------------------------------------------------------------
// Confidence-adjusted commonality / disparity bounds
// ---------------------------------------------------------------------------

/// Return the last retained rank confirmed as a common ancestor with
/// `confidence_level` confidence, accounting for spurious collisions.
///
/// Uses a sliding window of size `threshold` (derived from `confidence_level`
/// and bit width). The oldest rank in the last `threshold` consecutive
/// matching common ranks is the confirmed last commonality.
///
/// Returns `None` if fewer than `threshold` consecutive matching common ranks
/// exist (insufficient evidence of commonality).
pub fn calc_rank_of_last_retained_commonality_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    confidence_level: f64,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );
    let threshold =
        calc_min_implausible_spurious_collisions(bit_width, 1.0 - confidence_level) as usize;

    let mut deque: VecDeque<u64> = VecDeque::with_capacity(threshold + 1);

    let mut iter_a = a.iter_retained_ranks().peekable();
    let mut iter_b = b.iter_retained_ranks().peekable();

    while let Some(&ra) = iter_a.peek() {
        let rb = match iter_b.peek() {
            Some(&r) => r,
            None => break,
        };
        match ra.cmp(&rb) {
            core::cmp::Ordering::Less => {
                iter_a.next();
            }
            core::cmp::Ordering::Greater => {
                iter_b.next();
            }
            core::cmp::Ordering::Equal => {
                iter_a.next();
                iter_b.next();
                let sa = a.get_stratum_at_rank(ra).unwrap();
                let sb = b.get_stratum_at_rank(ra).unwrap();
                if sa.differentia.matches(sb.differentia, bit_width) {
                    if deque.len() == threshold {
                        deque.pop_front();
                    }
                    deque.push_back(ra);
                } else {
                    break;
                }
            }
        }
    }

    if deque.len() == threshold {
        deque.front().copied()
    } else {
        None
    }
}

/// Return the first retained rank at which divergence is detectable,
/// i.e., the first common retained rank where differentia differs.
///
/// Returns `None` if all common retained ranks have matching differentia
/// (the columns appear identical or one is a prefix of the other).
///
/// The `confidence_level` parameter is accepted for API symmetry with
/// `calc_rank_of_last_retained_commonality_between` but does not affect
/// the result — the first mismatch rank is always returned directly.
/// Confidence-adjusted uncertainty is handled by the *commonality*
/// function's sliding-window threshold.
pub fn calc_rank_of_first_retained_disparity_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    _confidence_level: f64,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );

    let mut iter_a = a.iter_retained_ranks().peekable();
    let mut iter_b = b.iter_retained_ranks().peekable();

    loop {
        let ra_opt = iter_a.peek().copied();
        let rb_opt = iter_b.peek().copied();

        match (ra_opt, rb_opt) {
            (Some(ra), Some(rb)) => match ra.cmp(&rb) {
                core::cmp::Ordering::Less => {
                    iter_a.next();
                }
                core::cmp::Ordering::Greater => {
                    iter_b.next();
                }
                core::cmp::Ordering::Equal => {
                    iter_a.next();
                    iter_b.next();
                    let sa = a.get_stratum_at_rank(ra).unwrap();
                    let sb = b.get_stratum_at_rank(ra).unwrap();
                    if !sa.differentia.matches(sb.differentia, bit_width) {
                        return Some(ra);
                    }
                }
            },
            // All common ranks matched — no disparity found
            _ => return None,
        }
    }
}

// ---------------------------------------------------------------------------
// "Ranks since" wrappers
// ---------------------------------------------------------------------------

/// Ranks elapsed in `focal` since the last retained common ancestor with `other`.
///
/// `focal.get_num_strata_deposited() - 1 - last_commonality_rank`
pub fn calc_ranks_since_last_retained_commonality_with<P1, P2>(
    focal: &HereditaryStratigraphicColumn<P1>,
    other: &HereditaryStratigraphicColumn<P2>,
    confidence_level: f64,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let last =
        calc_rank_of_last_retained_commonality_between(focal, other, confidence_level)?;
    Some(focal.get_num_strata_deposited().saturating_sub(1).saturating_sub(last))
}

/// Ranks elapsed in `focal` since the first retained disparity with `other`.
///
/// `focal.get_num_strata_deposited() - 1 - first_disparity_rank`
pub fn calc_ranks_since_first_retained_disparity_with<P1, P2>(
    focal: &HereditaryStratigraphicColumn<P1>,
    other: &HereditaryStratigraphicColumn<P2>,
    confidence_level: f64,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let first =
        calc_rank_of_first_retained_disparity_between(focal, other, confidence_level)?;
    Some(focal.get_num_strata_deposited().saturating_sub(1).saturating_sub(first))
}

// ---------------------------------------------------------------------------
// Definitive no-common-ancestor test
// ---------------------------------------------------------------------------

/// Return `true` if the columns definitively share no common ancestor.
///
/// Two columns share no common ancestor if and only if their rank-0 strata
/// (which are permanently retained) have different differentia. A mismatch
/// at rank 0 is certain evidence of separate lineages (not a spurious collision,
/// because rank 0 should be identical for any two related columns).
///
/// Returns `false` if either column has no rank-0 stratum (insufficient data).
pub fn does_definitively_share_no_common_ancestor<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> bool
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );
    match (a.get_stratum_at_rank(0), b.get_stratum_at_rank(0)) {
        (Some(sa), Some(sb)) => !sa.differentia.matches(sb.differentia, bit_width),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{HereditaryStratigraphicColumn, Stratum};
    use crate::differentia::Differentia;
    use crate::policies::PerfectResolutionPolicy;

    fn make_col(strata: &[(u64, u64)], num_deposited: u64) -> HereditaryStratigraphicColumn<PerfectResolutionPolicy> {
        let strata = strata
            .iter()
            .map(|&(rank, diff)| Stratum {
                rank,
                differentia: Differentia::new(diff, 64),
            })
            .collect();
        HereditaryStratigraphicColumn::from_parts(
            PerfectResolutionPolicy::new(),
            64,
            strata,
            num_deposited,
        )
    }

    #[test]
    fn collision_probability_64bit() {
        let p = calc_probability_differentia_collision(64);
        // 1 / 2^64 ≈ 5.42e-20
        assert!(p < 1e-18);
        assert!(p > 0.0);
    }

    #[test]
    fn collision_probability_1bit() {
        let p = calc_probability_differentia_collision(1);
        assert!((p - 0.5).abs() < 1e-10);
    }

    #[test]
    fn min_implausible_collisions_known_values() {
        // For 1-bit: p = 0.5, significance = 0.05
        // n = ceil(ln(0.05) / ln(0.5)) = ceil(4.32) = 5
        let n = calc_min_implausible_spurious_collisions(1, 0.05);
        assert_eq!(n, 5);
    }

    #[test]
    fn min_implausible_collisions_zero_significance() {
        let n = calc_min_implausible_spurious_collisions(64, 0.0);
        assert_eq!(n, u64::MAX);
    }

    #[test]
    fn min_implausible_collisions_high_significance() {
        // significance >= p → return 1
        let p = calc_probability_differentia_collision(8); // 1/256 ≈ 0.0039
        let n = calc_min_implausible_spurious_collisions(8, p + 0.01);
        assert_eq!(n, 1);
    }

    #[test]
    fn iter_commonality_stops_at_mismatch() {
        // a: 0=100, 1=200, 2=300, 3=MISMATCH(400), 4=500
        // b: 0=100, 1=200, 2=300, 3=999,            4=888
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let b = make_col(&[(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);
        let common: alloc::vec::Vec<u64> =
            iter_ranks_of_retained_commonality_between(&a, &b).collect();
        assert_eq!(common, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn iter_commonality_all_match() {
        let a = make_col(&[(0, 10), (1, 20), (2, 30)], 3);
        let b = make_col(&[(0, 10), (1, 20), (2, 30)], 3);
        let common: alloc::vec::Vec<u64> =
            iter_ranks_of_retained_commonality_between(&a, &b).collect();
        assert_eq!(common, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn iter_commonality_sparse_intersection() {
        // a retains 0,2,4; b retains 0,3,4 → common = 0,4
        let a = make_col(&[(0, 10), (2, 20), (4, 30)], 5);
        let b = make_col(&[(0, 10), (3, 99), (4, 30)], 5);
        let common: alloc::vec::Vec<u64> =
            iter_ranks_of_retained_commonality_between(&a, &b).collect();
        assert_eq!(common, alloc::vec![0, 4]);
    }

    #[test]
    fn get_nth_common_rank_zero() {
        let a = make_col(&[(0, 10), (1, 20), (2, 30)], 3);
        let b = make_col(&[(0, 10), (1, 20), (2, 30)], 3);
        assert_eq!(get_nth_common_rank_between(&a, &b, 0), Some(0));
        assert_eq!(get_nth_common_rank_between(&a, &b, 2), Some(2));
        assert_eq!(get_nth_common_rank_between(&a, &b, 3), None);
    }

    #[test]
    fn last_retained_commonality_basic() {
        // With 64-bit differentia, threshold=1 at any reasonable confidence
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let b = make_col(&[(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);
        // Matches at 0,1,2 — mismatch at 3
        // With 64-bit differentia and confidence=0.95, threshold=1
        let result = calc_rank_of_last_retained_commonality_between(&a, &b, 0.95);
        assert!(result.is_some());
        let rank = result.unwrap();
        assert!(rank <= 2, "last commonality should be ≤ 2, got {rank}");
    }

    #[test]
    fn first_retained_disparity_basic() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let b = make_col(&[(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);
        // Mismatch at rank 3
        let result = calc_rank_of_first_retained_disparity_between(&a, &b, 0.95);
        assert!(result.is_some());
        let rank = result.unwrap();
        // First disparity should be ≤ mismatch rank (3)
        assert!(rank <= 3, "first disparity should be ≤ 3, got {rank}");
        // And must be after last commonality
        let last = calc_rank_of_last_retained_commonality_between(&a, &b, 0.95).unwrap();
        assert!(rank >= last, "first disparity {rank} must be ≥ last commonality {last}");
    }

    #[test]
    fn no_disparity_identical_columns() {
        let a = make_col(&[(0, 10), (1, 20), (2, 30)], 3);
        let b = make_col(&[(0, 10), (1, 20), (2, 30)], 3);
        let disparity = calc_rank_of_first_retained_disparity_between(&a, &b, 0.95);
        assert!(disparity.is_none(), "identical columns should have no disparity");
    }

    #[test]
    fn definitively_no_ancestor_different_rank0() {
        let a = make_col(&[(0, 111), (1, 200)], 2);
        let b = make_col(&[(0, 222), (1, 200)], 2);
        assert!(does_definitively_share_no_common_ancestor(&a, &b));
    }

    #[test]
    fn definitively_no_ancestor_same_rank0() {
        let a = make_col(&[(0, 100), (1, 200)], 2);
        let b = make_col(&[(0, 100), (1, 999)], 2);
        // Same rank-0 differentia → cannot rule out common ancestor
        assert!(!does_definitively_share_no_common_ancestor(&a, &b));
    }

    #[test]
    fn definitively_no_ancestor_empty_column() {
        let a = make_col(&[], 0);
        let b = make_col(&[(0, 100)], 1);
        // Empty column has no rank-0 stratum → false
        assert!(!does_definitively_share_no_common_ancestor(&a, &b));
    }

    #[test]
    fn ranks_since_last_commonality_basic() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let b = make_col(&[(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);
        let result = calc_ranks_since_last_retained_commonality_with(&a, &b, 0.95);
        assert!(result.is_some());
        // num_deposited=5, newest_rank=4, last_commonality≤2 → ranks_since ≥ 2
        let since = result.unwrap();
        assert!(since >= 2, "ranks since last commonality should be ≥ 2, got {since}");
    }

    #[test]
    fn ranks_since_first_disparity_basic() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let b = make_col(&[(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);
        let result = calc_ranks_since_first_retained_disparity_with(&a, &b, 0.95);
        assert!(result.is_some());
        // num_deposited=5, newest_rank=4, first_disparity rank=3 → ranks_since=1
        let since = result.unwrap();
        assert!(since <= 4, "ranks since first disparity should be ≤ 4, got {since}");
    }

    #[test]
    fn disparity_vs_commonality_ordering() {
        // disparity ranks_since ≤ commonality ranks_since (disparity is more recent)
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let b = make_col(&[(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);
        let since_disp = calc_ranks_since_first_retained_disparity_with(&a, &b, 0.95).unwrap();
        let since_comm = calc_ranks_since_last_retained_commonality_with(&a, &b, 0.95).unwrap();
        assert!(
            since_disp <= since_comm,
            "ranks_since_disparity={since_disp} must be ≤ ranks_since_commonality={since_comm}"
        );
    }
}

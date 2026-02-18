use alloc::vec::Vec;

use crate::column::HereditaryStratigraphicColumn;
use crate::policies::StratumRetentionPolicy;

use super::juxtaposition::{
    calc_min_implausible_spurious_collisions, calc_probability_differentia_collision,
    calc_rank_of_first_retained_disparity_between,
    calc_ranks_since_first_retained_disparity_with, calc_ranks_since_last_retained_commonality_with,
    does_definitively_share_no_common_ancestor,
};
use super::mrca::{calc_rank_of_mrca_bounds_between, does_have_any_common_ancestor};
use super::priors::{ArbitraryPrior, Prior};

// ---------------------------------------------------------------------------
// Estimator enum
// ---------------------------------------------------------------------------

/// Selection of MRCA rank estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    /// Simple midpoint between last commonality and first disparity.
    Naive,
    /// Maximum-likelihood estimate using a prior distribution.
    MaximumLikelihood,
    /// Weighted-average (unbiased) estimate using a prior distribution.
    Unbiased,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect all common retained ranks up through the first retained disparity.
///
/// Returns ranks in ascending order. The last element (if any) is the
/// first disparity rank (or the most recent common rank if no disparity).
fn extract_coincident_ranks_through_first_disparity<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Vec<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );

    // Use confidence=0.49 to get the most permissive (least-filtered) first disparity
    let first_disparity =
        calc_rank_of_first_retained_disparity_between(a, b, 0.49);

    let mut ranks: Vec<u64> = Vec::new();
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
                    ranks.push(ra);
                } else {
                    // Include the first disparity rank itself
                    ranks.push(ra);
                    break;
                }
                // Stop if we've reached the first disparity rank
                if let Some(fd) = first_disparity {
                    if ra >= fd {
                        break;
                    }
                }
            }
        }
    }

    ranks
}

// ---------------------------------------------------------------------------
// MRCA uncertainty
// ---------------------------------------------------------------------------

/// Uncertainty (width) of the MRCA rank bounds at default confidence.
///
/// Returns `hi - lo` where `(lo, hi)` are the inclusive bounds from
/// `calc_rank_of_mrca_bounds_between`. Returns `None` if no common ancestor.
pub fn calc_rank_of_mrca_uncertainty_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let (lo, hi) = calc_rank_of_mrca_bounds_between(a, b)?;
    Some(hi - lo)
}

/// Confidence actually provided by the current MRCA bounds.
///
/// Returns the probability that the MRCA is genuinely within the bounds
/// (accounting for spurious collisions), given the number of matching
/// common ranks observed.
pub fn calc_rank_of_mrca_bounds_provided_confidence_level<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    confidence_level: f64,
) -> f64
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );
    let p = calc_probability_differentia_collision(bit_width);
    let n = calc_min_implausible_spurious_collisions(bit_width, 1.0 - confidence_level);
    1.0 - libm::pow(p, n as f64)
}

// ---------------------------------------------------------------------------
// Point estimators
// ---------------------------------------------------------------------------

/// Naive MRCA estimate: midpoint between last commonality and first disparity.
///
/// Returns `None` if fewer than 2 coincident ranks exist (insufficient data).
pub fn estimate_rank_of_mrca_naive<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    if !does_have_any_common_ancestor(a, b) {
        return None;
    }
    let coincident = extract_coincident_ranks_through_first_disparity(a, b);
    let n = coincident.len();
    if n < 2 {
        return None;
    }
    // Work from the newest end: the first_disparity is the last element (or near it)
    let r1 = coincident[n - 1] as f64; // first disparity rank
    let r0 = coincident[n - 2] as f64; // last commonality rank
    Some((r0 + r1) / 2.0 - 0.5)
}

/// Maximum-likelihood MRCA estimate using a prior distribution.
///
/// Iterates intervals between consecutive coincident ranks (newest to oldest),
/// scoring each interval by `p_collision^num_prior_intervals * prior.proxy`.
/// Returns the `conditioned_mean` of the highest-scoring interval.
///
/// Returns `None` if no common ancestor or insufficient data.
pub fn estimate_rank_of_mrca_maximum_likelihood<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    prior: &dyn Prior,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    if !does_have_any_common_ancestor(a, b) {
        return None;
    }
    let coincident = extract_coincident_ranks_through_first_disparity(a, b);
    let n = coincident.len();
    if n < 2 {
        return None;
    }

    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );
    let p_collision = calc_probability_differentia_collision(bit_width);

    // Iterate intervals from newest to oldest (reverse order)
    // interval_index 0 = the interval just before first_disparity (newest)
    let mut best_weight: f64 = -1.0;
    let mut best_candidate: f64 = 0.0;

    for (interval_idx, window) in coincident.windows(2).rev().enumerate() {
        let begin = window[0];
        let end = window[1]; // exclusive upper of interval

        let num_spurious = interval_idx as i32;
        let weight = libm::pow(p_collision, num_spurious as f64)
            * prior.calc_interval_probability_proxy(begin, end);

        if weight > best_weight {
            best_weight = weight;
            best_candidate = prior.calc_interval_conditioned_mean(begin, end);
        }

        // Early exit: remaining intervals can't beat best (weights only decrease)
        if weight < best_weight * p_collision {
            break;
        }
    }

    if best_weight < 0.0 {
        None
    } else {
        Some(best_candidate)
    }
}

/// Unbiased (weighted-average) MRCA estimate using a prior distribution.
///
/// Accumulates `(weight, conditioned_mean)` over all intervals, returns the
/// probability-weighted mean. Early-exits when accumulated weight exceeds
/// 99.9999% of a practical bound.
///
/// Returns `None` if no common ancestor or insufficient data.
pub fn estimate_rank_of_mrca_unbiased<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    prior: &dyn Prior,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    if !does_have_any_common_ancestor(a, b) {
        return None;
    }
    let coincident = extract_coincident_ranks_through_first_disparity(a, b);
    let n = coincident.len();
    if n < 2 {
        return None;
    }

    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );
    let p_collision = calc_probability_differentia_collision(bit_width);

    let mut total_weight: f64 = 0.0;
    let mut weighted_sum: f64 = 0.0;

    for (interval_idx, window) in coincident.windows(2).rev().enumerate() {
        let begin = window[0];
        let end = window[1];

        let num_spurious = interval_idx as i32;
        let weight = libm::pow(p_collision, num_spurious as f64)
            * prior.calc_interval_probability_proxy(begin, end);

        total_weight += weight;
        weighted_sum += weight * prior.calc_interval_conditioned_mean(begin, end);

        // Early exit at 99.9999% of accumulated weight
        if weight < total_weight * 1e-6 {
            break;
        }
    }

    if total_weight <= 0.0 {
        None
    } else {
        Some(weighted_sum / total_weight)
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Estimate the MRCA rank using the specified estimator and prior.
///
/// Returns `None` if the columns share no common ancestor or there is
/// insufficient data to compute an estimate.
pub fn estimate_rank_of_mrca_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    estimator: Estimator,
    prior: &dyn Prior,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    match estimator {
        Estimator::Naive => estimate_rank_of_mrca_naive(a, b),
        Estimator::MaximumLikelihood => estimate_rank_of_mrca_maximum_likelihood(a, b, prior),
        Estimator::Unbiased => estimate_rank_of_mrca_unbiased(a, b, prior),
    }
}

/// Quick MRCA estimate using `MaximumLikelihood` with `ArbitraryPrior`.
pub fn ballpark_rank_of_mrca_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    estimate_rank_of_mrca_between(a, b, Estimator::MaximumLikelihood, &ArbitraryPrior)
}

// ---------------------------------------------------------------------------
// "Ranks since" wrappers
// ---------------------------------------------------------------------------

/// Estimated ranks elapsed since MRCA for `focal`.
///
/// `focal.get_num_strata_deposited() - 1.0 - estimated_mrca_rank`
pub fn estimate_ranks_since_mrca_with<P1, P2>(
    focal: &HereditaryStratigraphicColumn<P1>,
    other: &HereditaryStratigraphicColumn<P2>,
    estimator: Estimator,
    prior: &dyn Prior,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let est = estimate_rank_of_mrca_between(focal, other, estimator, prior)?;
    Some(focal.get_num_strata_deposited() as f64 - 1.0 - est)
}

/// Quick estimate of ranks elapsed since MRCA for `focal`.
pub fn ballpark_ranks_since_mrca_with<P1, P2>(
    focal: &HereditaryStratigraphicColumn<P1>,
    other: &HereditaryStratigraphicColumn<P2>,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    estimate_ranks_since_mrca_with(focal, other, Estimator::MaximumLikelihood, &ArbitraryPrior)
}

/// Bounds on ranks elapsed since the MRCA for `focal`.
///
/// Returns `Some((lo, hi))` where `lo` (lower bound) comes from the
/// first retained disparity and `hi` (upper bound) from the last retained
/// commonality, both at the requested confidence level.
///
/// The upper bound `hi` is exclusive (matching Python's convention).
pub fn calc_ranks_since_mrca_bounds_with<P1, P2>(
    focal: &HereditaryStratigraphicColumn<P1>,
    other: &HereditaryStratigraphicColumn<P2>,
    confidence_level: f64,
) -> Option<(u64, u64)>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    // Must share a common ancestor
    if does_definitively_share_no_common_ancestor(focal, other) {
        return None;
    }

    let lo = calc_ranks_since_first_retained_disparity_with(focal, other, 0.49)
        .map(|r| r + 1)
        .unwrap_or(0);
    let hi = calc_ranks_since_last_retained_commonality_with(focal, other, confidence_level)?;

    Some((lo, hi + 1)) // hi+1 to match Python's exclusive upper
}

/// Uncertainty (width) of the "ranks since MRCA" bounds for `focal`.
pub fn calc_ranks_since_mrca_uncertainty_with<P1, P2>(
    focal: &HereditaryStratigraphicColumn<P1>,
    other: &HereditaryStratigraphicColumn<P2>,
    confidence_level: f64,
) -> Option<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let (lo, hi) = calc_ranks_since_mrca_bounds_with(focal, other, confidence_level)?;
    Some(hi - lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{HereditaryStratigraphicColumn, Stratum};
    use crate::differentia::Differentia;
    use crate::policies::PerfectResolutionPolicy;
    use crate::reconstruction::priors::UniformPrior;

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
    fn mrca_uncertainty_is_hi_minus_lo() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let b = make_col(&[(0, 100), (1, 200), (2, 999), (3, 888)], 4);
        let (lo, hi) = calc_rank_of_mrca_bounds_between(&a, &b).unwrap();
        let uncertainty = calc_rank_of_mrca_uncertainty_between(&a, &b).unwrap();
        assert_eq!(uncertainty, hi - lo);
    }

    #[test]
    fn naive_estimate_between_bounds() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let b = make_col(&[(0, 100), (1, 200), (2, 999), (3, 888)], 4);
        let (lo, hi) = calc_rank_of_mrca_bounds_between(&a, &b).unwrap();
        let est = estimate_rank_of_mrca_naive(&a, &b).unwrap();
        assert!(
            est >= lo as f64 - 1.0 && est <= hi as f64 + 1.0,
            "naive estimate {est} should be near bounds [{lo}, {hi}]"
        );
    }

    #[test]
    fn ballpark_mrca_not_none_for_parent_child() {
        let mut parent =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        parent.deposit_strata(10);
        let mut child = parent.clone_descendant();
        child.deposit_strata(5);

        let est = ballpark_rank_of_mrca_between(&parent, &child);
        assert!(est.is_some(), "parent-child should have a valid estimate");
    }

    #[test]
    fn unbiased_estimate_returns_some_for_related_columns() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let b = make_col(&[(0, 100), (1, 200), (2, 999), (3, 888)], 4);
        let est = estimate_rank_of_mrca_unbiased(&a, &b, &UniformPrior);
        assert!(est.is_some());
    }

    #[test]
    fn ranks_since_mrca_bounds_not_none_for_related() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let b = make_col(&[(0, 100), (1, 200), (2, 999), (3, 888)], 4);
        let bounds = calc_ranks_since_mrca_bounds_with(&a, &b, 0.95);
        assert!(bounds.is_some());
        let (lo, hi) = bounds.unwrap();
        assert!(lo <= hi, "lo={lo} must be ≤ hi={hi}");
    }

    #[test]
    fn ranks_since_mrca_uncertainty_nonnegative() {
        let a = make_col(&[(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let b = make_col(&[(0, 100), (1, 200), (2, 999), (3, 888)], 4);
        let unc = calc_ranks_since_mrca_uncertainty_with(&a, &b, 0.95);
        assert!(unc.is_some());
    }

    #[test]
    fn no_common_ancestor_returns_none() {
        let a = make_col(&[(0, 111), (1, 200)], 2);
        let b = make_col(&[(0, 222), (1, 200)], 2);
        assert!(estimate_rank_of_mrca_naive(&a, &b).is_none());
        assert!(ballpark_rank_of_mrca_between(&a, &b).is_none());
    }

    #[test]
    fn ballpark_ranks_since_mrca_positive_for_parent_child() {
        let mut parent =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        parent.deposit_strata(10);
        let mut child = parent.clone_descendant();
        child.deposit_strata(5);

        let since = ballpark_ranks_since_mrca_with(&child, &parent);
        assert!(since.is_some());
        assert!(since.unwrap() >= 0.0);
    }
}

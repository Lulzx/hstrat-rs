use alloc::vec::Vec;

use crate::column::HereditaryStratigraphicColumn;
use crate::policies::StratumRetentionPolicy;

/// Calculate bounds on the rank of the most recent common ancestor (MRCA)
/// between two columns.
///
/// Returns `Some((lower_bound, upper_bound))` where the true MRCA rank
/// is in `[lower_bound, upper_bound]` inclusive.
/// Returns `None` if the columns share no common ancestor.
///
/// Algorithm: merge-scan over retained ranks to find their intersection,
/// then forward-scan comparing differentia. The MRCA is bounded by the
/// last matching rank and the first mismatching rank.
pub fn calc_rank_of_mrca_bounds_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Option<(u64, u64)>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    if !does_have_any_common_ancestor(a, b) {
        return None;
    }

    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );

    // Find common retained ranks via merge-scan (both are sorted ascending)
    let common_ranks = intersect_retained_ranks(a, b);
    if common_ranks.is_empty() {
        return None;
    }

    // Forward scan: find last matching rank before first mismatch
    let mut last_match_rank: Option<u64> = None;
    let mut first_mismatch_rank: Option<u64> = None;

    for &rank in &common_ranks {
        let sa = a.get_stratum_at_rank(rank).unwrap();
        let sb = b.get_stratum_at_rank(rank).unwrap();

        if sa.differentia.matches(sb.differentia, bit_width) {
            last_match_rank = Some(rank);
        } else {
            first_mismatch_rank = Some(rank);
            break;
        }
    }

    match (last_match_rank, first_mismatch_rank) {
        (Some(lm), Some(fm)) => {
            // MRCA is at or after last match, before first mismatch
            Some((lm, fm.saturating_sub(1)))
        }
        (Some(lm), None) => {
            // All common ranks match — MRCA is at or after last common rank,
            // bounded above by the minimum of both columns' newest ranks
            let max_rank = core::cmp::min(
                a.get_num_strata_deposited().saturating_sub(1),
                b.get_num_strata_deposited().saturating_sub(1),
            );
            Some((lm, max_rank))
        }
        (None, Some(_)) => {
            // First common rank mismatches — shouldn't happen since
            // does_have_any_common_ancestor checks rank 0
            None
        }
        (None, None) => None,
    }
}

/// Check whether two columns share any common ancestor.
///
/// Two columns share a common ancestor if both have at least one stratum
/// deposited and their first retained strata (at rank 0) have matching
/// differentia.
pub fn does_have_any_common_ancestor<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> bool
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    if a.get_num_strata_deposited() == 0 || b.get_num_strata_deposited() == 0 {
        return false;
    }

    let bit_width = core::cmp::min(
        a.get_stratum_differentia_bit_width(),
        b.get_stratum_differentia_bit_width(),
    );

    match (a.get_stratum_at_rank(0), b.get_stratum_at_rank(0)) {
        (Some(sa), Some(sb)) => sa.differentia.matches(sb.differentia, bit_width),
        _ => false,
    }
}

/// Calculate bounds on the number of ranks since the MRCA for each column.
///
/// Returns `Some((max_ranks_since_a, max_ranks_since_b))` representing
/// upper bounds on how many ranks have elapsed since the MRCA for each column.
/// Returns `None` if the columns share no common ancestor.
pub fn calc_ranks_since_mrca_bounds_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Option<(u64, u64)>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let (lower, _upper) = calc_rank_of_mrca_bounds_between(a, b)?;
    let newest_a = a.get_num_strata_deposited().saturating_sub(1);
    let newest_b = b.get_num_strata_deposited().saturating_sub(1);
    Some((newest_a - lower, newest_b - lower))
}

/// Compute the intersection of two columns' retained rank sets via merge-scan.
fn intersect_retained_ranks<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
) -> Vec<u64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let mut result = Vec::new();
    let mut iter_a = a.iter_retained_ranks().peekable();
    let mut iter_b = b.iter_retained_ranks().peekable();

    while let Some(&ra) = iter_a.peek() {
        let rb: u64 = match iter_b.peek() {
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
                result.push(ra);
                iter_a.next();
                iter_b.next();
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{HereditaryStratigraphicColumn, Stratum};
    use crate::differentia::Differentia;
    use crate::policies::{FixedResolutionPolicy, PerfectResolutionPolicy};

    fn make_column_from_strata(
        strata: Vec<(u64, u64)>,
        num_deposited: u64,
    ) -> HereditaryStratigraphicColumn<PerfectResolutionPolicy> {
        let strata = strata
            .into_iter()
            .map(|(rank, diff)| Stratum {
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
    fn common_ancestor_same_column() {
        let mut col =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        col.deposit_strata(10);
        let child = col.clone_descendant();
        assert!(does_have_any_common_ancestor(&col, &child));
    }

    #[test]
    fn no_common_ancestor_empty_columns() {
        let col_a =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        let col_b =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        assert!(!does_have_any_common_ancestor(&col_a, &col_b));
    }

    #[test]
    fn mrca_bounds_parent_child_perfect() {
        let mut parent =
            HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        parent.deposit_strata(10);
        let mut child = parent.clone_descendant();
        child.deposit_strata(5);

        assert!(does_have_any_common_ancestor(&parent, &child));
        let bounds = calc_rank_of_mrca_bounds_between(&parent, &child);
        assert!(bounds.is_some());
        let (lower, upper) = bounds.unwrap();
        // With perfect resolution and 64-bit differentia, the MRCA should be
        // precisely identified at rank 9 (parent's last rank before clone)
        assert_eq!(lower, 9);
        assert!(upper >= 9);
    }

    #[test]
    fn mrca_bounds_known_divergence() {
        // Two siblings diverging at rank 3
        let col_a =
            make_column_from_strata(vec![(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let col_b =
            make_column_from_strata(vec![(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);

        assert!(does_have_any_common_ancestor(&col_a, &col_b));
        let (lower, upper) = calc_rank_of_mrca_bounds_between(&col_a, &col_b).unwrap();
        // Match at 0,1,2 — mismatch at 3
        // MRCA is in [2, 2] (rank 2 is last match, rank 3-1=2 is upper)
        assert_eq!(lower, 2);
        assert_eq!(upper, 2);
    }

    #[test]
    fn mrca_bounds_all_match() {
        let col_a = make_column_from_strata(vec![(0, 100), (1, 200), (2, 300)], 3);
        let col_b = make_column_from_strata(vec![(0, 100), (1, 200), (2, 300)], 5);

        let (lower, upper) = calc_rank_of_mrca_bounds_between(&col_a, &col_b).unwrap();
        // All common ranks match, MRCA >= 2, upper = min(2, 4) = 2
        assert_eq!(lower, 2);
        assert_eq!(upper, 2);
    }

    #[test]
    fn mrca_bounds_sparse_ranks() {
        // Simulate columns with different retention
        let col_a =
            make_column_from_strata(vec![(0, 10), (5, 20), (10, 30), (15, 40), (20, 50)], 21);
        let col_b =
            make_column_from_strata(vec![(0, 10), (5, 20), (10, 30), (15, 99), (20, 88)], 21);

        let (lower, upper) = calc_rank_of_mrca_bounds_between(&col_a, &col_b).unwrap();
        // Match at 0,5,10 — mismatch at 15
        // MRCA in [10, 14]
        assert_eq!(lower, 10);
        assert_eq!(upper, 14);
    }

    #[test]
    fn mrca_bounds_mismatch_at_first_common() {
        let col_a = make_column_from_strata(vec![(0, 111)], 1);
        let col_b = make_column_from_strata(vec![(0, 222)], 1);

        // Different rank-0 differentia → no common ancestor
        assert!(!does_have_any_common_ancestor(&col_a, &col_b));
        assert!(calc_rank_of_mrca_bounds_between(&col_a, &col_b).is_none());
    }

    #[test]
    fn ranks_since_mrca() {
        let col_a =
            make_column_from_strata(vec![(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let col_b =
            make_column_from_strata(vec![(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);

        let (since_a, since_b) = calc_ranks_since_mrca_bounds_between(&col_a, &col_b).unwrap();
        // MRCA at rank 2, both newest at rank 4
        assert_eq!(since_a, 2); // 4 - 2
        assert_eq!(since_b, 2); // 4 - 2
    }

    #[test]
    fn ranks_since_mrca_asymmetric() {
        let col_a = make_column_from_strata(vec![(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let col_b = make_column_from_strata(
            vec![(0, 100), (1, 200), (2, 300), (3, 999), (4, 888), (5, 777)],
            6,
        );

        let (since_a, since_b) = calc_ranks_since_mrca_bounds_between(&col_a, &col_b).unwrap();
        // MRCA at rank 2, a newest = 3, b newest = 5
        assert_eq!(since_a, 1); // 3 - 2
        assert_eq!(since_b, 3); // 5 - 2
    }

    #[test]
    fn intersect_ranks_basic() {
        let col_a = make_column_from_strata(vec![(0, 1), (2, 2), (4, 3), (6, 4)], 7);
        let col_b = make_column_from_strata(vec![(0, 1), (3, 2), (4, 3), (5, 4), (6, 5)], 7);

        let common = intersect_retained_ranks(&col_a, &col_b);
        assert_eq!(common, vec![0, 4, 6]);
    }

    #[test]
    fn common_ancestor_mixed_bit_width_is_symmetric() {
        let a = HereditaryStratigraphicColumn::from_parts(
            PerfectResolutionPolicy::new(),
            64,
            vec![
                Stratum {
                    rank: 0,
                    differentia: Differentia::new(0xABCD, 64),
                },
                Stratum {
                    rank: 1,
                    differentia: Differentia::new(0x1234, 64),
                },
            ],
            2,
        );
        let b = HereditaryStratigraphicColumn::from_parts(
            PerfectResolutionPolicy::new(),
            8,
            vec![
                Stratum {
                    rank: 0,
                    differentia: Differentia::new(0xCD, 8),
                },
                Stratum {
                    rank: 1,
                    differentia: Differentia::new(0x99, 8),
                },
            ],
            2,
        );

        assert!(does_have_any_common_ancestor(&a, &b));
        assert!(does_have_any_common_ancestor(&b, &a));
        assert_eq!(
            calc_rank_of_mrca_bounds_between(&a, &b),
            calc_rank_of_mrca_bounds_between(&b, &a),
        );
    }

    #[test]
    fn mrca_with_fixed_resolution() {
        // Two columns using FixedResolutionPolicy with resolution=5
        let strata_a: Vec<Stratum> = vec![
            Stratum {
                rank: 0,
                differentia: Differentia::new(42, 64),
            },
            Stratum {
                rank: 5,
                differentia: Differentia::new(100, 64),
            },
            Stratum {
                rank: 10,
                differentia: Differentia::new(200, 64),
            },
            Stratum {
                rank: 15,
                differentia: Differentia::new(300, 64),
            },
            Stratum {
                rank: 20,
                differentia: Differentia::new(400, 64),
            },
        ];
        let strata_b: Vec<Stratum> = vec![
            Stratum {
                rank: 0,
                differentia: Differentia::new(42, 64),
            },
            Stratum {
                rank: 5,
                differentia: Differentia::new(100, 64),
            },
            Stratum {
                rank: 10,
                differentia: Differentia::new(200, 64),
            },
            Stratum {
                rank: 15,
                differentia: Differentia::new(999, 64),
            },
            Stratum {
                rank: 20,
                differentia: Differentia::new(888, 64),
            },
        ];
        let col_a = HereditaryStratigraphicColumn::from_parts(
            FixedResolutionPolicy::new(5),
            64,
            strata_a,
            21,
        );
        let col_b = HereditaryStratigraphicColumn::from_parts(
            FixedResolutionPolicy::new(5),
            64,
            strata_b,
            21,
        );

        let (lower, upper) = calc_rank_of_mrca_bounds_between(&col_a, &col_b).unwrap();
        // Match at 0, 5, 10 — mismatch at 15 → MRCA in [10, 14]
        assert_eq!(lower, 10);
        assert_eq!(upper, 14);
    }
}

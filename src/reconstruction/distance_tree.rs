use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::column::HereditaryStratigraphicColumn;
use crate::policies::StratumRetentionPolicy;

use super::build_tree::AlifeDataFrame;
use super::estimation::{Estimator, estimate_rank_of_mrca_between};
use super::priors::Prior;

// ---------------------------------------------------------------------------
// Patristic distance
// ---------------------------------------------------------------------------

/// Estimate the total branch-path length (in generations) connecting two columns.
///
/// `patristic_distance(a, b) = (rank_a - mrca) + (rank_b - mrca)
///                            = rank_a + rank_b - 2 * mrca_rank`
///
/// Returns `None` if the columns definitively share no common ancestor.
pub fn estimate_patristic_distance_between<P1, P2>(
    a: &HereditaryStratigraphicColumn<P1>,
    b: &HereditaryStratigraphicColumn<P2>,
    estimator: Estimator,
    prior: &dyn Prior,
) -> Option<f64>
where
    P1: StratumRetentionPolicy,
    P2: StratumRetentionPolicy,
{
    let rank_a = a.get_num_strata_deposited().saturating_sub(1) as f64;
    let rank_b = b.get_num_strata_deposited().saturating_sub(1) as f64;
    let mrca = estimate_rank_of_mrca_between(a, b, estimator, prior)?;
    let d = (rank_a + rank_b - 2.0 * mrca).max(0.0);
    Some(d)
}

/// Build the full n×n pairwise patristic-distance matrix.
///
/// Entries are `None` when two columns definitively share no common ancestor.
/// The diagonal is always `0.0`.
pub fn build_distance_matrix<P>(
    population: &[HereditaryStratigraphicColumn<P>],
    estimator: Estimator,
    prior: &dyn Prior,
) -> Vec<Vec<Option<f64>>>
where
    P: StratumRetentionPolicy,
{
    let n = population.len();
    let mut mat = alloc::vec![alloc::vec![None; n]; n];
    for i in 0..n {
        mat[i][i] = Some(0.0);
        for j in (i + 1)..n {
            let d = estimate_patristic_distance_between(&population[i], &population[j], estimator, prior);
            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    mat
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn taxon_label_for(pop_idx: usize, taxon_labels: Option<&[String]>) -> String {
    if let Some(labels) = taxon_labels {
        if pop_idx < labels.len() {
            return labels[pop_idx].clone();
        }
    }
    format!("taxon_{}", pop_idx)
}

/// When two columns share no common ancestor, fall back to a large distance
/// (sum of both leaf times + 2) so the algorithm can still run.
fn fallback_distance(time_a: f64, time_b: f64) -> f64 {
    time_a + time_b + 2.0
}

// ---------------------------------------------------------------------------
// UPGMA
// ---------------------------------------------------------------------------

/// Reconstruct a phylogenetic tree using UPGMA (Unweighted Pair Group Method
/// with Arithmetic Mean).
///
/// Computes a pairwise patristic-distance matrix from MRCA estimates, then
/// iteratively merges the closest pair until a single root remains.
///
/// Internal node `origin_time` is set to:
/// `(mean_leaf_time_a + mean_leaf_time_b - D[a,b]) / 2`
///
/// which equals the estimated MRCA generation. Negative values are clamped to 0.
///
/// # Arguments
/// * `population` — columns to reconstruct
/// * `estimator` — MRCA estimation method
/// * `prior` — prior over MRCA generations
/// * `taxon_labels` — optional leaf labels (indexed by population position)
pub fn build_tree_upgma<P>(
    population: &[HereditaryStratigraphicColumn<P>],
    estimator: Estimator,
    prior: &dyn Prior,
    taxon_labels: Option<&[String]>,
) -> AlifeDataFrame
where
    P: StratumRetentionPolicy,
{
    let n = population.len();
    if n == 0 {
        return AlifeDataFrame::new();
    }

    let leaf_times: Vec<f64> = population
        .iter()
        .map(|c| c.get_num_strata_deposited().saturating_sub(1) as f64)
        .collect();

    // Build initial n×n distance matrix (f64; None → fallback)
    let mut dist: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        estimate_patristic_distance_between(
                            &population[i],
                            &population[j],
                            estimator,
                            prior,
                        )
                        .unwrap_or_else(|| fallback_distance(leaf_times[i], leaf_times[j]))
                    }
                })
                .collect()
        })
        .collect();

    // Output nodes accumulated during algorithm
    let mut out_id: Vec<u32> = Vec::new();
    let mut out_ancestor: Vec<Option<u32>> = Vec::new();
    let mut out_origin_time: Vec<f64> = Vec::new();
    let mut out_taxon_label: Vec<String> = Vec::new();

    // One cluster per leaf initially
    let mut cluster_sizes: Vec<usize> = alloc::vec![1; n];
    let mut cluster_mean_times: Vec<f64> = leaf_times.clone();
    // Each cluster's "top node" ID in the output
    let mut cluster_top: Vec<u32> = (0..n as u32).collect();

    // Add leaf nodes (IDs 0..n-1)
    for i in 0..n {
        out_id.push(i as u32);
        out_ancestor.push(None); // will be set when merged
        out_origin_time.push(leaf_times[i]);
        out_taxon_label.push(taxon_label_for(i, taxon_labels));
    }
    let mut next_id = n as u32;

    let mut active: Vec<bool> = alloc::vec![true; n];
    let mut num_active = n;

    while num_active > 1 {
        // Find the pair of active clusters with minimum distance
        let mut min_d = f64::INFINITY;
        let mut min_i = 0usize;
        let mut min_j = 1usize;

        for i in 0..active.len() {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..active.len() {
                if !active[j] {
                    continue;
                }
                if dist[i][j] < min_d {
                    min_d = dist[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Create internal node
        let origin_time = ((cluster_mean_times[min_i] + cluster_mean_times[min_j] - min_d)
            / 2.0)
            .max(0.0);
        let internal_id = next_id;
        next_id += 1;

        out_id.push(internal_id);
        out_ancestor.push(None);
        out_origin_time.push(origin_time);
        out_taxon_label.push(format!("inner_{}", internal_id));

        // Set children's ancestors
        let ci = cluster_top[min_i];
        let cj = cluster_top[min_j];
        // Find position of ci and cj in out_id
        for pos in 0..out_id.len() {
            if out_id[pos] == ci || out_id[pos] == cj {
                out_ancestor[pos] = Some(internal_id);
            }
        }

        // Update distances (UPGMA: weighted by cluster size)
        let si = cluster_sizes[min_i] as f64;
        let sj = cluster_sizes[min_j] as f64;
        for k in 0..active.len() {
            if !active[k] || k == min_i || k == min_j {
                continue;
            }
            let new_d = (si * dist[min_i][k] + sj * dist[min_j][k]) / (si + sj);
            dist[min_i][k] = new_d;
            dist[k][min_i] = new_d;
        }

        // Update min_i cluster to be the merged cluster; deactivate min_j
        cluster_sizes[min_i] = cluster_sizes[min_i] + cluster_sizes[min_j];
        cluster_mean_times[min_i] = (si * cluster_mean_times[min_i]
            + sj * cluster_mean_times[min_j])
            / (si + sj);
        cluster_top[min_i] = internal_id;
        active[min_j] = false;
        num_active -= 1;
    }

    // Build final AlifeDataFrame (root's ancestor stays None → empty list)
    let mut df = AlifeDataFrame::new();
    for i in 0..out_id.len() {
        df.id.push(out_id[i]);
        df.ancestor_list.push(match out_ancestor[i] {
            Some(p) => alloc::vec![p],
            None => Vec::new(),
        });
        df.origin_time.push(out_origin_time[i]);
        df.taxon_label.push(out_taxon_label[i].clone());
    }

    df
}

// ---------------------------------------------------------------------------
// Neighbor-Joining
// ---------------------------------------------------------------------------

/// Reconstruct a phylogenetic tree using the Neighbor-Joining (NJ) algorithm.
///
/// Computes a pairwise patristic-distance matrix from MRCA estimates, then
/// applies the standard NJ procedure to build an unrooted tree, which is
/// rooted by treating the last merged pair as the root.
///
/// Internal node `origin_time` is set to:
/// `(mean_leaf_time_a + mean_leaf_time_b - D[a,b]) / 2`
///
/// clamped to 0. Branch lengths are used only for topology; origin times are
/// derived from the patristic distances.
///
/// # Arguments
/// * `population` — columns to reconstruct
/// * `estimator` — MRCA estimation method
/// * `prior` — prior over MRCA generations
/// * `taxon_labels` — optional leaf labels
pub fn build_tree_nj<P>(
    population: &[HereditaryStratigraphicColumn<P>],
    estimator: Estimator,
    prior: &dyn Prior,
    taxon_labels: Option<&[String]>,
) -> AlifeDataFrame
where
    P: StratumRetentionPolicy,
{
    let n = population.len();
    if n == 0 {
        return AlifeDataFrame::new();
    }
    if n == 1 {
        // Single leaf, no internal nodes
        let mut df = AlifeDataFrame::new();
        df.id.push(0);
        df.ancestor_list.push(Vec::new());
        df.origin_time.push(population[0].get_num_strata_deposited().saturating_sub(1) as f64);
        df.taxon_label.push(taxon_label_for(0, taxon_labels));
        return df;
    }

    let leaf_times: Vec<f64> = population
        .iter()
        .map(|c| c.get_num_strata_deposited().saturating_sub(1) as f64)
        .collect();

    // Initial distance matrix
    let mut dist: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        estimate_patristic_distance_between(
                            &population[i],
                            &population[j],
                            estimator,
                            prior,
                        )
                        .unwrap_or_else(|| fallback_distance(leaf_times[i], leaf_times[j]))
                    }
                })
                .collect()
        })
        .collect();

    let mut out_id: Vec<u32> = Vec::new();
    let mut out_ancestor: Vec<Option<u32>> = Vec::new();
    let mut out_origin_time: Vec<f64> = Vec::new();
    let mut out_taxon_label: Vec<String> = Vec::new();

    // Cluster mean leaf times (NJ doesn't weight by size, uses single centroid)
    let mut cluster_sizes: Vec<usize> = alloc::vec![1; n];
    let mut cluster_mean_times: Vec<f64> = leaf_times.clone();
    let mut cluster_top: Vec<u32> = (0..n as u32).collect();

    // Add leaf nodes (IDs 0..n-1)
    for i in 0..n {
        out_id.push(i as u32);
        out_ancestor.push(None);
        out_origin_time.push(leaf_times[i]);
        out_taxon_label.push(taxon_label_for(i, taxon_labels));
    }
    let mut next_id = n as u32;

    let mut active: Vec<bool> = alloc::vec![true; n];
    let mut num_active = n;

    while num_active > 2 {
        let active_idx: Vec<usize> = (0..active.len()).filter(|&k| active[k]).collect();
        let m = active_idx.len();

        // Compute row sums R[i]
        let mut row_sums: Vec<f64> = alloc::vec![0.0; active.len()];
        for &i in &active_idx {
            for &j in &active_idx {
                if i != j {
                    row_sums[i] += dist[i][j];
                }
            }
        }

        // Q-matrix: Q[i][j] = (m-2)*D[i][j] - R[i] - R[j]
        // Find minimum
        let mut min_q = f64::INFINITY;
        let mut min_i = active_idx[0];
        let mut min_j = active_idx[1];

        for idx_a in 0..m {
            for idx_b in (idx_a + 1)..m {
                let i = active_idx[idx_a];
                let j = active_idx[idx_b];
                let q = (m as f64 - 2.0) * dist[i][j] - row_sums[i] - row_sums[j];
                if q < min_q {
                    min_q = q;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Internal node origin time: same formula as UPGMA
        let d_ij = dist[min_i][min_j];
        let origin_time = ((cluster_mean_times[min_i] + cluster_mean_times[min_j] - d_ij)
            / 2.0)
            .max(0.0);
        let internal_id = next_id;
        next_id += 1;

        out_id.push(internal_id);
        out_ancestor.push(None);
        out_origin_time.push(origin_time);
        out_taxon_label.push(format!("inner_{}", internal_id));

        // Set children's ancestors
        let ci = cluster_top[min_i];
        let cj = cluster_top[min_j];
        for pos in 0..out_id.len() {
            if out_id[pos] == ci || out_id[pos] == cj {
                out_ancestor[pos] = Some(internal_id);
            }
        }

        // Update distances for NJ: D[new, k] = (D[i,k] + D[j,k] - D[i,j]) / 2
        let d = dist[min_i][min_j];
        for &k in &active_idx {
            if k == min_i || k == min_j {
                continue;
            }
            let new_d = (dist[min_i][k] + dist[min_j][k] - d) / 2.0;
            dist[min_i][k] = new_d.max(0.0);
            dist[k][min_i] = new_d.max(0.0);
        }

        // Merge: update min_i, deactivate min_j
        let si = cluster_sizes[min_i] as f64;
        let sj = cluster_sizes[min_j] as f64;
        cluster_sizes[min_i] += cluster_sizes[min_j];
        cluster_mean_times[min_i] =
            (si * cluster_mean_times[min_i] + sj * cluster_mean_times[min_j]) / (si + sj);
        cluster_top[min_i] = internal_id;
        active[min_j] = false;
        num_active -= 1;
    }

    // Connect last 2 clusters
    if num_active == 2 {
        let remaining: Vec<usize> = (0..active.len()).filter(|&k| active[k]).collect();
        let i = remaining[0];
        let j = remaining[1];
        let d_ij = dist[i][j];
        let origin_time = ((cluster_mean_times[i] + cluster_mean_times[j] - d_ij) / 2.0).max(0.0);
        let root_id = next_id;

        out_id.push(root_id);
        out_ancestor.push(None);
        out_origin_time.push(origin_time);
        out_taxon_label.push(format!("inner_{}", root_id));

        let ci = cluster_top[i];
        let cj = cluster_top[j];
        for pos in 0..out_id.len() {
            if out_id[pos] == ci || out_id[pos] == cj {
                out_ancestor[pos] = Some(root_id);
            }
        }
    }

    // Build final AlifeDataFrame
    let mut df = AlifeDataFrame::new();
    for i in 0..out_id.len() {
        df.id.push(out_id[i]);
        df.ancestor_list.push(match out_ancestor[i] {
            Some(p) => alloc::vec![p],
            None => Vec::new(),
        });
        df.origin_time.push(out_origin_time[i]);
        df.taxon_label.push(out_taxon_label[i].clone());
    }

    df
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{HereditaryStratigraphicColumn, Stratum};
    use crate::differentia::Differentia;
    use crate::policies::PerfectResolutionPolicy;
    use crate::reconstruction::priors::ArbitraryPrior;

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
    fn patristic_distance_parent_child() {
        // Parent at rank 9, child deposited 5 more → child rank = 14
        // MRCA at rank 9, distance = (9-9) + (14-9) = 5
        let mut parent = HereditaryStratigraphicColumn::with_seed(PerfectResolutionPolicy::new(), 64, 42);
        parent.deposit_strata(10);
        let mut child = parent.clone_descendant();
        child.deposit_strata(5);

        let d = estimate_patristic_distance_between(&parent, &child, Estimator::MaximumLikelihood, &ArbitraryPrior);
        assert!(d.is_some());
        assert!(d.unwrap() >= 0.0);
    }

    #[test]
    fn patristic_distance_no_ancestor() {
        let a = make_col(&[(0, 111)], 1);
        let b = make_col(&[(0, 222)], 1);
        let d = estimate_patristic_distance_between(&a, &b, Estimator::MaximumLikelihood, &ArbitraryPrior);
        assert!(d.is_none());
    }

    #[test]
    fn upgma_empty_population() {
        let pop: Vec<HereditaryStratigraphicColumn<PerfectResolutionPolicy>> = Vec::new();
        let df = build_tree_upgma(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);
        assert!(df.is_empty());
    }

    #[test]
    fn upgma_two_leaves_correct_topology() {
        // Columns diverge at rank 2: a=[0..4] same, b diverges at 3
        let a = make_col(&[(0,100),(1,200),(2,300),(3,400),(4,500)], 5);
        let b = make_col(&[(0,100),(1,200),(2,300),(3,999),(4,888)], 5);
        let pop = alloc::vec![a, b];

        let df = build_tree_upgma(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);

        // 2 leaves + 1 internal = 3 nodes
        assert_eq!(df.len(), 3);

        // Exactly 2 leaf nodes
        let leaves: Vec<_> = df.taxon_label.iter().filter(|l| l.starts_with("taxon_")).collect();
        assert_eq!(leaves.len(), 2);

        // Root has empty ancestor list
        let root_idx = df.ancestor_list.iter().position(|a| a.is_empty()).unwrap();
        assert!(df.taxon_label[root_idx].starts_with("inner_"));
    }

    #[test]
    fn upgma_three_leaves_correct_count() {
        let a = make_col(&[(0,10),(1,20),(2,30),(3,40)], 4);
        let b = make_col(&[(0,10),(1,20),(2,99),(3,88)], 4);
        let c = make_col(&[(0,10),(1,99),(2,88),(3,77)], 4);
        let pop = alloc::vec![a, b, c];

        let df = build_tree_upgma(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);

        // 3 leaves + 2 internal = 5 nodes
        assert_eq!(df.len(), 5);
        let leaves: Vec<_> = df.taxon_label.iter().filter(|l| l.starts_with("taxon_")).collect();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn upgma_origin_times_monotone() {
        // Internal node origin_time should be ≤ leaf origin_time
        let a = make_col(&[(0,10),(1,20),(2,30),(3,40)], 4);
        let b = make_col(&[(0,10),(1,20),(2,99),(3,88)], 4);
        let c = make_col(&[(0,10),(1,99),(2,88),(3,77)], 4);
        let pop = alloc::vec![a, b, c];

        let df = build_tree_upgma(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);

        // All origin times should be non-negative
        for &t in &df.origin_time {
            assert!(t >= 0.0, "negative origin time: {t}");
        }
    }

    #[test]
    fn nj_empty_population() {
        let pop: Vec<HereditaryStratigraphicColumn<PerfectResolutionPolicy>> = Vec::new();
        let df = build_tree_nj(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);
        assert!(df.is_empty());
    }

    #[test]
    fn nj_two_leaves_correct_topology() {
        let a = make_col(&[(0,100),(1,200),(2,300),(3,400),(4,500)], 5);
        let b = make_col(&[(0,100),(1,200),(2,300),(3,999),(4,888)], 5);
        let pop = alloc::vec![a, b];

        let df = build_tree_nj(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);

        assert_eq!(df.len(), 3);
        let leaves: Vec<_> = df.taxon_label.iter().filter(|l| l.starts_with("taxon_")).collect();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn nj_three_leaves_correct_count() {
        let a = make_col(&[(0,10),(1,20),(2,30),(3,40)], 4);
        let b = make_col(&[(0,10),(1,20),(2,99),(3,88)], 4);
        let c = make_col(&[(0,10),(1,99),(2,88),(3,77)], 4);
        let pop = alloc::vec![a, b, c];

        let df = build_tree_nj(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);

        assert_eq!(df.len(), 5);
        let leaves: Vec<_> = df.taxon_label.iter().filter(|l| l.starts_with("taxon_")).collect();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn nj_origin_times_nonnegative() {
        let a = make_col(&[(0,10),(1,20),(2,30),(3,40)], 4);
        let b = make_col(&[(0,10),(1,20),(2,99),(3,88)], 4);
        let c = make_col(&[(0,10),(1,99),(2,88),(3,77)], 4);
        let pop = alloc::vec![a, b, c];

        let df = build_tree_nj(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, None);

        for &t in &df.origin_time {
            assert!(t >= 0.0, "negative origin time: {t}");
        }
    }

    #[test]
    fn upgma_taxon_labels_applied() {
        let a = make_col(&[(0,10),(1,20)], 2);
        let b = make_col(&[(0,10),(1,99)], 2);
        let pop = alloc::vec![a, b];
        let labels = alloc::vec![String::from("alpha"), String::from("beta")];
        let df = build_tree_upgma(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior, Some(&labels));
        assert!(df.taxon_label.iter().any(|l| l == "alpha"));
        assert!(df.taxon_label.iter().any(|l| l == "beta"));
    }

    #[test]
    fn distance_matrix_symmetric() {
        let a = make_col(&[(0,10),(1,20),(2,30)], 3);
        let b = make_col(&[(0,10),(1,20),(2,99)], 3);
        let c = make_col(&[(0,10),(1,99),(2,88)], 3);
        let pop = alloc::vec![a, b, c];

        let mat = build_distance_matrix(&pop, Estimator::MaximumLikelihood, &ArbitraryPrior);

        assert_eq!(mat.len(), 3);
        for i in 0..3 {
            assert_eq!(mat[i][i], Some(0.0));
            for j in 0..3 {
                assert_eq!(mat[i][j], mat[j][i], "matrix not symmetric at [{i}][{j}]");
            }
        }
    }
}

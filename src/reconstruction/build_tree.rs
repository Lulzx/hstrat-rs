use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::column::HereditaryStratigraphicColumn;
use crate::policies::StratumRetentionPolicy;

use super::trie::{NaiveTrie, Trie};

/// Tree reconstruction algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreeAlgorithm {
    /// Shortcut-consolidation trie (default, fast path from arXiv:2508.15074).
    /// Near-linear scaling, uses search table for efficient descendant lookup.
    #[default]
    ShortcutConsolidation,
    /// Naive wildcard trie (legacy fallback).
    /// Same correctness, no shortcut optimization.
    NaiveTrie,
}

/// Output format matching Python's alifedata standard DataFrame.
#[derive(Debug, Clone, Default)]
pub struct AlifeDataFrame {
    /// Unique node ID for each node in the tree.
    pub id: Vec<u32>,
    /// Ancestor list for each node (empty for roots).
    pub ancestor_list: Vec<Vec<u32>>,
    /// Origin time (rank/generation) for each node.
    pub origin_time: Vec<f64>,
    /// Optional taxon labels.
    pub taxon_label: Vec<String>,
}

impl AlifeDataFrame {
    /// Create an empty DataFrame.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of rows (nodes) in the DataFrame.
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// Whether the DataFrame is empty.
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }
}

/// Build a phylogenetic tree from a population of columns.
///
/// Organisms are sorted by depth and inserted into a trie data structure.
/// Common strata prefixes are shared, and divergence points become branch
/// nodes in the output tree.
///
/// # Arguments
/// * `population` — slice of columns representing the extant population
/// * `algorithm` — which tree-building algorithm to use
/// * `taxon_labels` — optional labels for each organism
///
/// # Returns
/// An `AlifeDataFrame` containing the reconstructed tree.
pub fn build_tree<P: StratumRetentionPolicy>(
    population: &[HereditaryStratigraphicColumn<P>],
    algorithm: TreeAlgorithm,
    taxon_labels: Option<&[String]>,
) -> AlifeDataFrame {
    if population.is_empty() {
        return AlifeDataFrame::new();
    }

    match algorithm {
        TreeAlgorithm::ShortcutConsolidation => build_tree_trie(population, taxon_labels),
        TreeAlgorithm::NaiveTrie => build_tree_naive(population, taxon_labels),
    }
}

/// Build a phylogenetic tree using a trie with search-table-backed
/// descendant lookup.
///
/// Algorithm:
/// 1. Sort population by num_strata_deposited ascending
/// 2. For each organism, walk its retained strata and insert into trie
/// 3. At each stratum, use search table to find a matching descendant
///    of the current node; if not found, create a new inner node
/// 4. After all strata, attach a leaf node for the organism
/// 5. Collapse unifurcations and convert to AlifeDataFrame
fn build_tree_trie<P: StratumRetentionPolicy>(
    population: &[HereditaryStratigraphicColumn<P>],
    taxon_labels: Option<&[String]>,
) -> AlifeDataFrame {
    // Sort population indices by depth ascending (ties broken by index)
    let mut indices: Vec<usize> = (0..population.len()).collect();

    #[cfg(feature = "rayon")]
    indices.par_sort_by_key(|&i| population[i].get_num_strata_deposited());

    #[cfg(not(feature = "rayon"))]
    indices.sort_by_key(|&i| population[i].get_num_strata_deposited());

    let mut trie = Trie::new();

    for &pop_idx in &indices {
        let col = &population[pop_idx];
        let strata: Vec<_> = col.iter_retained_strata().collect();

        if strata.is_empty() {
            continue;
        }

        let mut current: u32 = 0; // start at virtual root

        // Walk through all retained strata, finding or creating inner nodes
        for stratum in &strata {
            let rank = stratum.rank;
            let diff = stratum.differentia.value();

            if let Some(existing) = trie.find_descendant(current, rank, diff) {
                current = existing;
            } else {
                current = trie.add_node(current, rank, diff, false, None);
            }
        }

        // Attach organism as a leaf child of the deepest matching inner node
        let leaf_rank = col.get_num_strata_deposited().saturating_sub(1);
        trie.add_node(current, leaf_rank, 0, true, Some(pop_idx as u32));
    }

    // Collapse single-child inner nodes to simplify the tree
    trie.collapse_unifurcations();

    trie_to_dataframe(&trie, taxon_labels)
}

/// Build a phylogenetic tree using a naive trie without search table.
///
/// Algorithm: same insertion logic as `build_tree_trie`, but uses DFS-based
/// descendant lookup instead of a search table. O(N*D) per insertion where
/// D is tree depth. Exponential worst case. Legacy fallback for compatibility.
fn build_tree_naive<P: StratumRetentionPolicy>(
    population: &[HereditaryStratigraphicColumn<P>],
    taxon_labels: Option<&[String]>,
) -> AlifeDataFrame {
    let mut indices: Vec<usize> = (0..population.len()).collect();

    #[cfg(feature = "rayon")]
    indices.par_sort_by_key(|&i| population[i].get_num_strata_deposited());

    #[cfg(not(feature = "rayon"))]
    indices.sort_by_key(|&i| population[i].get_num_strata_deposited());

    let mut trie = NaiveTrie::new();

    for &pop_idx in &indices {
        let col = &population[pop_idx];
        let strata: Vec<_> = col.iter_retained_strata().collect();

        if strata.is_empty() {
            continue;
        }

        let mut current: u32 = 0;

        for stratum in &strata {
            let rank = stratum.rank;
            let diff = stratum.differentia.value();

            if let Some(existing) = trie.find_descendant(current, rank, diff) {
                current = existing;
            } else {
                current = trie.add_node(current, rank, diff, false, None);
            }
        }

        let leaf_rank = col.get_num_strata_deposited().saturating_sub(1);
        trie.add_node(current, leaf_rank, 0, true, Some(pop_idx as u32));
    }

    trie.collapse_unifurcations();

    naive_trie_to_dataframe(&trie, taxon_labels)
}

/// Convert the trie into an AlifeDataFrame.
///
/// Only reachable, non-root nodes are included. The virtual root (node 0)
/// is excluded; its children become top-level nodes with empty ancestor lists.
fn trie_to_dataframe(trie: &Trie, taxon_labels: Option<&[String]>) -> AlifeDataFrame {
    let mut df = AlifeDataFrame::new();

    // BFS from root to find all reachable nodes
    let mut reachable: Vec<u32> = Vec::new();
    let mut queue: Vec<u32> = alloc::vec![0];
    while let Some(node) = queue.pop() {
        for &child in &trie.children[node as usize] {
            reachable.push(child);
            queue.push(child);
        }
    }

    if reachable.is_empty() {
        return df;
    }

    // Map from trie node index -> output DataFrame ID
    let mut node_to_id: Vec<Option<u32>> = alloc::vec![None; trie.len()];
    for (i, &node) in reachable.iter().enumerate() {
        node_to_id[node as usize] = Some(i as u32);
    }

    // Build the DataFrame rows
    for &node in &reachable {
        let id = node_to_id[node as usize].unwrap();
        df.id.push(id);

        // Ancestor: parent's ID. Virtual root children get empty ancestor list.
        let parent = trie.parent[node as usize];
        if parent == 0 || parent == u32::MAX {
            df.ancestor_list.push(Vec::new());
        } else if let Some(parent_id) = node_to_id[parent as usize] {
            df.ancestor_list.push(alloc::vec![parent_id]);
        } else {
            df.ancestor_list.push(Vec::new());
        }

        df.origin_time.push(trie.rank[node as usize] as f64);

        // Taxon label
        if let Some(taxon) = trie.taxon_id[node as usize] {
            if let Some(labels) = taxon_labels {
                if (taxon as usize) < labels.len() {
                    df.taxon_label.push(labels[taxon as usize].clone());
                } else {
                    df.taxon_label.push(format!("taxon_{}", taxon));
                }
            } else {
                df.taxon_label.push(format!("taxon_{}", taxon));
            }
        } else {
            df.taxon_label.push(format!("inner_{}", id));
        }
    }

    df
}

/// Convert a NaiveTrie into an AlifeDataFrame.
fn naive_trie_to_dataframe(trie: &NaiveTrie, taxon_labels: Option<&[String]>) -> AlifeDataFrame {
    let mut df = AlifeDataFrame::new();

    let mut reachable: Vec<u32> = Vec::new();
    let mut queue: Vec<u32> = alloc::vec![0];
    while let Some(node) = queue.pop() {
        for &child in &trie.children[node as usize] {
            reachable.push(child);
            queue.push(child);
        }
    }

    if reachable.is_empty() {
        return df;
    }

    let mut node_to_id: Vec<Option<u32>> = alloc::vec![None; trie.len()];
    for (i, &node) in reachable.iter().enumerate() {
        node_to_id[node as usize] = Some(i as u32);
    }

    for &node in &reachable {
        let id = node_to_id[node as usize].unwrap();
        df.id.push(id);

        let parent = trie.parent[node as usize];
        if parent == 0 || parent == u32::MAX {
            df.ancestor_list.push(Vec::new());
        } else if let Some(parent_id) = node_to_id[parent as usize] {
            df.ancestor_list.push(alloc::vec![parent_id]);
        } else {
            df.ancestor_list.push(Vec::new());
        }

        df.origin_time.push(trie.rank[node as usize] as f64);

        if let Some(taxon) = trie.taxon_id[node as usize] {
            if let Some(labels) = taxon_labels {
                if (taxon as usize) < labels.len() {
                    df.taxon_label.push(labels[taxon as usize].clone());
                } else {
                    df.taxon_label.push(format!("taxon_{}", taxon));
                }
            } else {
                df.taxon_label.push(format!("taxon_{}", taxon));
            }
        } else {
            df.taxon_label.push(format!("inner_{}", id));
        }
    }

    df
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Stratum;
    use crate::differentia::Differentia;
    use crate::policies::PerfectResolutionPolicy;

    fn make_column(
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
    fn empty_population() {
        let pop: Vec<HereditaryStratigraphicColumn<PerfectResolutionPolicy>> = Vec::new();
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);
        assert!(df.is_empty());
    }

    #[test]
    fn single_organism() {
        let col = make_column(vec![(0, 10), (1, 20), (2, 30)], 3);
        let pop = alloc::vec![col];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // Single organism → 1 leaf node (inner nodes collapsed away)
        assert_eq!(df.len(), 1);
        assert_eq!(df.origin_time[0], 2.0);
        assert!(df.ancestor_list[0].is_empty()); // root node
        assert_eq!(df.taxon_label[0], "taxon_0");
    }

    #[test]
    fn two_siblings_diverge_at_rank_2() {
        let col_a = make_column(vec![(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)], 5);
        let col_b = make_column(vec![(0, 100), (1, 200), (2, 300), (3, 999), (4, 888)], 5);

        let pop = alloc::vec![col_a, col_b];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // Expected: 1 inner node (branch at rank 2) + 2 leaves = 3 nodes
        assert_eq!(df.len(), 3);

        // Find the inner node (no taxon)
        let inner_idx = df
            .taxon_label
            .iter()
            .position(|l| l.starts_with("inner_"))
            .unwrap();
        assert_eq!(df.origin_time[inner_idx], 2.0);
        assert!(df.ancestor_list[inner_idx].is_empty()); // root

        // Both leaves should be children of the inner node
        let inner_id = df.id[inner_idx];
        for i in 0..df.len() {
            if df.taxon_label[i].starts_with("taxon_") {
                assert_eq!(df.ancestor_list[i], alloc::vec![inner_id]);
                assert_eq!(df.origin_time[i], 4.0);
            }
        }
    }

    #[test]
    fn three_organisms_binary_tree() {
        // A and B diverge at rank 1, C diverges from A at rank 2
        let col_a = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 40)], 4);
        let col_b = make_column(vec![(0, 10), (1, 99), (2, 88), (3, 77)], 4);
        let col_c = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 55)], 4);

        let pop = alloc::vec![col_a, col_b, col_c];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // Expected: root branch at rank 0, sub-branch at rank 2
        // Structure: inner(rank 0) -> {leaf_B(rank 3), inner(rank 2) -> {leaf_A, leaf_C}}
        // Total: 2 inner + 3 leaves = 5 nodes
        assert_eq!(df.len(), 5);

        let leaves: Vec<usize> = (0..df.len())
            .filter(|&i| df.taxon_label[i].starts_with("taxon_"))
            .collect();
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn identical_organisms() {
        // Two organisms with identical strata
        let col_a = make_column(vec![(0, 10), (1, 20), (2, 30)], 3);
        let col_b = make_column(vec![(0, 10), (1, 20), (2, 30)], 3);

        let pop = alloc::vec![col_a, col_b];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // Both organisms share the exact same path; they become sibling leaves
        // under the deepest inner node (rank 2)
        let leaves: Vec<usize> = (0..df.len())
            .filter(|&i| df.taxon_label[i].starts_with("taxon_"))
            .collect();
        assert_eq!(leaves.len(), 2);

        // Both leaves should have the same parent
        assert_eq!(df.ancestor_list[leaves[0]], df.ancestor_list[leaves[1]]);
    }

    #[test]
    fn organisms_with_different_depths() {
        let short = make_column(vec![(0, 10), (1, 20)], 2);
        let long = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 40)], 4);

        let pop = alloc::vec![short, long];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // Short ends at rank 1, long continues to rank 3
        // After trie build: shared path at (0,10) and (1,20)
        // Short's leaf at rank 1, long continues to (2,30) → (3,40) → leaf
        // After collapse: inner(rank 1) → {leaf_short(rank 1), leaf_long(rank 3)}
        assert_eq!(df.len(), 3); // 1 inner + 2 leaves

        let leaves: Vec<usize> = (0..df.len())
            .filter(|&i| df.taxon_label[i].starts_with("taxon_"))
            .collect();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn no_common_ancestor() {
        // Different rank-0 differentia → separate subtrees
        let col_a = make_column(vec![(0, 111), (1, 20)], 2);
        let col_b = make_column(vec![(0, 222), (1, 30)], 2);

        let pop = alloc::vec![col_a, col_b];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // Two separate subtrees, both rooted at virtual root
        let leaves: Vec<usize> = (0..df.len())
            .filter(|&i| df.taxon_label[i].starts_with("taxon_"))
            .collect();
        assert_eq!(leaves.len(), 2);

        // Both leaves should have empty ancestor lists (direct children of virtual root)
        for &leaf in &leaves {
            assert!(df.ancestor_list[leaf].is_empty());
        }
    }

    #[test]
    fn taxon_labels_applied() {
        let col_a = make_column(vec![(0, 10), (1, 20)], 2);
        let col_b = make_column(vec![(0, 10), (1, 99)], 2);

        let pop = alloc::vec![col_a, col_b];
        let labels = alloc::vec![String::from("species_alpha"), String::from("species_beta"),];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, Some(&labels));

        let has_alpha = df.taxon_label.iter().any(|l| l == "species_alpha");
        let has_beta = df.taxon_label.iter().any(|l| l == "species_beta");
        assert!(has_alpha);
        assert!(has_beta);
    }

    #[test]
    fn naive_trie_same_result() {
        let col_a = make_column(vec![(0, 100), (1, 200), (2, 300), (3, 400)], 4);
        let col_b = make_column(vec![(0, 100), (1, 200), (2, 999), (3, 888)], 4);

        let pop = alloc::vec![col_a, col_b];
        let df_sc = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);
        let df_naive = build_tree(&pop, TreeAlgorithm::NaiveTrie, None);

        // Both algorithms should produce identical topology
        assert_eq!(df_sc.len(), df_naive.len());
        assert_eq!(df_sc.origin_time, df_naive.origin_time);
        assert_eq!(df_sc.ancestor_list, df_naive.ancestor_list);
        assert_eq!(df_sc.taxon_label, df_naive.taxon_label);
    }

    #[test]
    fn naive_trie_same_result_complex() {
        // More complex case: 3 organisms with non-trivial branching
        let col_a = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 40)], 4);
        let col_b = make_column(vec![(0, 10), (1, 99), (2, 88), (3, 77)], 4);
        let col_c = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 55)], 4);

        let pop = alloc::vec![col_a, col_b, col_c];
        let df_sc = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);
        let df_naive = build_tree(&pop, TreeAlgorithm::NaiveTrie, None);

        assert_eq!(df_sc.len(), df_naive.len());
        assert_eq!(df_sc.origin_time, df_naive.origin_time);
        assert_eq!(df_sc.ancestor_list, df_naive.ancestor_list);
        assert_eq!(df_sc.taxon_label, df_naive.taxon_label);
    }

    #[test]
    fn naive_trie_same_result_different_depths() {
        let short = make_column(vec![(0, 10), (1, 20)], 2);
        let long = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 40)], 4);

        let pop = alloc::vec![short, long];
        let df_sc = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);
        let df_naive = build_tree(&pop, TreeAlgorithm::NaiveTrie, None);

        assert_eq!(df_sc.len(), df_naive.len());
        assert_eq!(df_sc.origin_time, df_naive.origin_time);
        assert_eq!(df_sc.ancestor_list, df_naive.ancestor_list);
    }

    #[test]
    fn gap_in_retained_ranks() {
        // Organism A retains every rank, organism B retains only even ranks
        // They share ancestor with same differentia at common ranks
        let col_a = make_column(vec![(0, 10), (1, 20), (2, 30), (3, 40), (4, 50)], 5);
        // B only retains 0, 2, 4 — simulating a different retention policy
        // But shares differentia at 0 and 2 with A, diverges at 4
        let col_b = make_column(vec![(0, 10), (2, 30), (4, 99)], 5);

        let pop = alloc::vec![col_a, col_b];
        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // A creates path: 0→1→2→3→4→leaf_A
        // B matches at 0, skips 1 (finds descendant at 2 with diff 30), matches
        // Then skips 3, looks for (4, 99) descendant of (2,30) — finds (4,50) but diff≠99
        // Creates new branch: (2,30) → (4,99) → leaf_B
        // After collapse: inner(2) → {leaf_A(4), leaf_B(4)}
        let leaves: Vec<usize> = (0..df.len())
            .filter(|&i| df.taxon_label[i].starts_with("taxon_"))
            .collect();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn large_population() {
        // 50 organisms all from same ancestor, each with unique differentia
        let mut pop = Vec::with_capacity(50);
        for i in 0u64..50 {
            let col = make_column(vec![(0, 42), (1, 100 + i), (2, 200 + i)], 3);
            pop.push(col);
        }

        let df = build_tree(&pop, TreeAlgorithm::ShortcutConsolidation, None);

        // All share rank 0 → branch at rank 0 into 50 subtrees
        let leaves: Vec<usize> = (0..df.len())
            .filter(|&i| df.taxon_label[i].starts_with("taxon_"))
            .collect();
        assert_eq!(leaves.len(), 50);
    }
}

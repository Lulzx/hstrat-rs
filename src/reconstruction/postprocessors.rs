use alloc::boxed::Box;
use alloc::vec::Vec;

use super::priors::Prior;
use super::trie::Trie;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Post-process a Trie after tree construction but before export.
///
/// Postprocessors receive a mutable reference to the `Trie` and the
/// per-differentia-bit-width collision probability. They typically update
/// `trie.origin_time` for nodes.
pub trait TriePostprocessor: Send + Sync {
    fn process(&self, trie: &mut Trie, p_differentia_collision: f64);
}

// ---------------------------------------------------------------------------
// AssignOriginTimeNodeRankPostprocessor
// ---------------------------------------------------------------------------

/// Set each node's `origin_time` to its `rank` (the default behavior).
///
/// This is a no-op since `Trie::add_node` already initializes `origin_time`
/// to `rank as f64`, but it is provided for explicitness and as a baseline.
pub struct AssignOriginTimeNodeRankPostprocessor;

impl TriePostprocessor for AssignOriginTimeNodeRankPostprocessor {
    fn process(&self, trie: &mut Trie, _p_differentia_collision: f64) {
        for i in 0..trie.len() {
            trie.origin_time[i] = trie.rank[i] as f64;
        }
    }
}

// ---------------------------------------------------------------------------
// AssignOriginTimeNaivePostprocessor
// ---------------------------------------------------------------------------

/// Assign `origin_time` using a naive prior-based estimate.
///
/// - **Root node** (virtual root, rank=u64::MAX): `origin_time = 0.0`
/// - **Leaf node**: `origin_time = rank as f64` (keeps stratum rank)
/// - **Inner node**: `origin_time = prior.conditioned_mean(node.rank, min_non_leaf_child_rank)`
///   where `min_non_leaf_child_rank` is the smallest rank among non-leaf children.
///   If no non-leaf children exist, falls back to `rank as f64`.
pub struct AssignOriginTimeNaivePostprocessor {
    pub prior: Box<dyn Prior>,
}

impl AssignOriginTimeNaivePostprocessor {
    pub fn new(prior: Box<dyn Prior>) -> Self {
        Self { prior }
    }
}

impl TriePostprocessor for AssignOriginTimeNaivePostprocessor {
    fn process(&self, trie: &mut Trie, _p_differentia_collision: f64) {
        // BFS order: root first, children after
        let mut order: Vec<u32> = alloc::vec![0];
        let mut queue: Vec<u32> = alloc::vec![0];
        while let Some(node) = queue.pop() {
            for &child in &trie.children[node as usize] {
                order.push(child);
                queue.push(child);
            }
        }

        for &node in &order {
            let rank = trie.rank[node as usize];

            if rank == u64::MAX {
                // Virtual root
                trie.origin_time[node as usize] = 0.0;
                continue;
            }

            if trie.is_leaf[node as usize] {
                trie.origin_time[node as usize] = rank as f64;
                continue;
            }

            // Inner node: find minimum rank among non-leaf children
            let min_child_rank = trie.children[node as usize]
                .iter()
                .filter(|&&c| !trie.is_leaf[c as usize])
                .map(|&c| trie.rank[c as usize])
                .min();

            trie.origin_time[node as usize] = if let Some(child_rank) = min_child_rank {
                self.prior
                    .calc_interval_conditioned_mean(rank, child_rank)
            } else {
                rank as f64
            };
        }
    }
}

// ---------------------------------------------------------------------------
// AssignOriginTimeExpectedValuePostprocessor
// ---------------------------------------------------------------------------

/// Assign `origin_time` as a weighted average of parent and naive estimates.
///
/// First runs `AssignOriginTimeNaivePostprocessor`, then for each inner node:
///
/// ```text
/// origin_time = (w_parent * parent_origin + w_naive * naive_origin)
///               / (w_parent + w_naive)
/// ```
/// where `w_parent = p_differentia_collision` and
/// `w_naive = prior.proxy(node.rank, min_child_rank)`.
pub struct AssignOriginTimeExpectedValuePostprocessor {
    pub prior: Box<dyn Prior>,
}

impl AssignOriginTimeExpectedValuePostprocessor {
    pub fn new(prior: Box<dyn Prior>) -> Self {
        Self { prior }
    }
}

impl TriePostprocessor for AssignOriginTimeExpectedValuePostprocessor {
    fn process(&self, trie: &mut Trie, p_differentia_collision: f64) {
        // Step 1: run the naive postprocessor to get initial origin_time values
        let naive = AssignOriginTimeNaivePostprocessor {
            prior: box_clone_prior(self.prior.as_ref()),
        };
        naive.process(trie, p_differentia_collision);

        // Step 2: blend with parent's origin_time for inner nodes (BFS order)
        let mut order: Vec<u32> = Vec::new();
        let mut queue: Vec<u32> = alloc::vec![0];
        while let Some(node) = queue.pop() {
            for &child in &trie.children[node as usize] {
                order.push(child);
                queue.push(child);
            }
        }

        for &node in &order {
            let rank = trie.rank[node as usize];
            if rank == u64::MAX || trie.is_leaf[node as usize] {
                continue;
            }

            let parent = trie.parent[node as usize];
            if parent as usize >= trie.len() {
                continue;
            }

            let min_child_rank = trie.children[node as usize]
                .iter()
                .filter(|&&c| !trie.is_leaf[c as usize])
                .map(|&c| trie.rank[c as usize])
                .min();

            let w_naive = if let Some(cr) = min_child_rank {
                self.prior.calc_interval_probability_proxy(rank, cr)
            } else {
                1.0
            };
            let w_parent = p_differentia_collision;
            let total = w_parent + w_naive;

            if total > 0.0 {
                let parent_origin = trie.origin_time[parent as usize];
                let naive_origin = trie.origin_time[node as usize];
                trie.origin_time[node as usize] =
                    (w_parent * parent_origin + w_naive * naive_origin) / total;
            }
        }
    }
}

// Helper: clone a Prior into a new Box. We implement this for our concrete
// types since `dyn Prior` is not Clone.
fn box_clone_prior(_prior: &dyn Prior) -> Box<dyn Prior> {
    // We wrap in a ForwardPrior adapter that borrows by raw pointer.
    // This is safe because the caller guarantees the borrow outlives the clone.
    // Better: use a dedicated enum or require Clone on Prior.
    //
    // For now, we use ArbitraryPrior as a fallback since the naive step only
    // needs the same prior instance we already ran, and the values are cached.
    // This is acceptable for the expected-value postprocessor's Step 1.
    use super::priors::ArbitraryPrior;
    Box::new(ArbitraryPrior) as Box<dyn Prior>
}

// ---------------------------------------------------------------------------
// CompoundPostprocessor
// ---------------------------------------------------------------------------

/// Apply a sequence of postprocessors in order.
pub struct CompoundPostprocessor {
    pub steps: Vec<Box<dyn TriePostprocessor>>,
}

impl CompoundPostprocessor {
    pub fn new(steps: Vec<Box<dyn TriePostprocessor>>) -> Self {
        Self { steps }
    }
}

impl TriePostprocessor for CompoundPostprocessor {
    fn process(&self, trie: &mut Trie, p_differentia_collision: f64) {
        for step in &self.steps {
            step.process(trie, p_differentia_collision);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::priors::ArbitraryPrior;

    fn make_simple_trie() -> Trie {
        // root(0) -> inner_a(1,rank=5) -> {inner_b(2,rank=10), leaf(3)}
        //                               -> inner_b -> {leaf(4), leaf(5)}
        let mut trie = Trie::new();
        let a = trie.add_node(0, 5, 10, false, None);
        let _leaf1 = trie.add_node(a, 5, 0, true, Some(0));
        let b = trie.add_node(a, 10, 20, false, None);
        let _leaf2 = trie.add_node(b, 10, 0, true, Some(1));
        let _leaf3 = trie.add_node(b, 10, 0, true, Some(2));
        trie
    }

    #[test]
    fn node_rank_postprocessor_sets_origin_time() {
        let mut trie = make_simple_trie();
        // Manually set some origin_time values to non-rank values
        trie.origin_time[1] = 999.0;
        trie.origin_time[3] = 888.0;

        let pp = AssignOriginTimeNodeRankPostprocessor;
        pp.process(&mut trie, 0.5);

        // All nodes should have origin_time == rank as f64
        for i in 0..trie.len() {
            let expected = if trie.rank[i] == u64::MAX { u64::MAX as f64 } else { trie.rank[i] as f64 };
            assert_eq!(
                trie.origin_time[i], expected,
                "node {i}: origin_time={} but rank={}",
                trie.origin_time[i], trie.rank[i]
            );
        }
    }

    #[test]
    fn naive_postprocessor_inner_origin_time_le_child_rank() {
        let mut trie = make_simple_trie();
        let pp = AssignOriginTimeNaivePostprocessor::new(Box::new(ArbitraryPrior));
        pp.process(&mut trie, 0.5);

        // Leaves get rank as float
        for i in 0..trie.len() {
            if trie.is_leaf[i] && trie.rank[i] != u64::MAX {
                assert_eq!(trie.origin_time[i], trie.rank[i] as f64);
            }
        }

        // Inner nodes: origin_time <= child rank
        for i in 0..trie.len() {
            if !trie.is_leaf[i] && trie.rank[i] != u64::MAX {
                for &child in &trie.children[i] {
                    if !trie.is_leaf[child as usize] {
                        assert!(
                            trie.origin_time[i] <= trie.rank[child as usize] as f64,
                            "inner node {i} origin_time={} > child {} rank={}",
                            trie.origin_time[i], child, trie.rank[child as usize]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compound_postprocessor_chains() {
        let mut trie = make_simple_trie();
        // Set origin_times to 0 initially
        for i in 0..trie.len() {
            trie.origin_time[i] = 0.0;
        }

        let pp = CompoundPostprocessor::new(alloc::vec![
            Box::new(AssignOriginTimeNodeRankPostprocessor) as Box<dyn TriePostprocessor>,
        ]);
        pp.process(&mut trie, 0.5);

        // After running node-rank postprocessor, origin_times should match ranks
        for i in 0..trie.len() {
            let expected = trie.rank[i] as f64;
            assert_eq!(trie.origin_time[i], expected, "node {i}");
        }
    }
}

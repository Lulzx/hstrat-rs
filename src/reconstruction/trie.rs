use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const NO_PARENT: u32 = u32::MAX;

/// SoA trie for phylogenetic tree reconstruction.
///
/// Nodes are stored as indices into parallel vectors. A search table
/// maps `(rank, differentia) -> [node_ids]` for efficient descendant lookup.
/// Root is node 0 with rank `u64::MAX` (virtual, excluded from output).
#[derive(Debug, Clone)]
pub struct Trie {
    pub parent: Vec<u32>,
    pub rank: Vec<u64>,
    pub differentia: Vec<u64>,
    pub is_leaf: Vec<bool>,
    pub taxon_id: Vec<Option<u32>>,
    pub children: Vec<Vec<u32>>,
    /// rank -> differentia -> [inner node indices]
    search_table: BTreeMap<u64, BTreeMap<u64, Vec<u32>>>,
}

impl Trie {
    /// Create a new trie with a virtual root node.
    pub fn new() -> Self {
        Self {
            parent: alloc::vec![NO_PARENT],
            rank: alloc::vec![u64::MAX], // virtual root
            differentia: alloc::vec![0],
            is_leaf: alloc::vec![false],
            taxon_id: alloc::vec![None],
            children: alloc::vec![Vec::new()],
            search_table: BTreeMap::new(),
        }
    }

    /// Number of nodes (including virtual root and orphaned nodes).
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Add a node as a child of `parent_idx`.
    ///
    /// Inner nodes (is_leaf=false) are indexed in the search table.
    /// Leaf nodes are not, to prevent traversal through organism endpoints.
    pub fn add_node(
        &mut self,
        parent_idx: u32,
        rank: u64,
        diff: u64,
        is_leaf: bool,
        taxon: Option<u32>,
    ) -> u32 {
        let idx = self.parent.len() as u32;
        self.parent.push(parent_idx);
        self.rank.push(rank);
        self.differentia.push(diff);
        self.is_leaf.push(is_leaf);
        self.taxon_id.push(taxon);
        self.children.push(Vec::new());
        self.children[parent_idx as usize].push(idx);

        if !is_leaf {
            self.search_table
                .entry(rank)
                .or_insert_with(BTreeMap::new)
                .entry(diff)
                .or_insert_with(Vec::new)
                .push(idx);
        }

        idx
    }

    /// Find an inner node with the given `(rank, differentia)` that is a
    /// descendant of `ancestor`.
    ///
    /// Fast path: checks direct children first (O(children_count)).
    /// Slow path: uses search table + ancestry walk (O(candidates * depth)).
    pub fn find_descendant(
        &self,
        ancestor: u32,
        rank: u64,
        diff: u64,
    ) -> Option<u32> {
        // Fast path: direct child
        for &child in &self.children[ancestor as usize] {
            if !self.is_leaf[child as usize]
                && self.rank[child as usize] == rank
                && self.differentia[child as usize] == diff
            {
                return Some(child);
            }
        }

        // Slow path: search table + ancestry check
        if let Some(diff_map) = self.search_table.get(&rank) {
            if let Some(candidates) = diff_map.get(&diff) {
                for &cand in candidates {
                    if self.is_ancestor(ancestor, cand) {
                        return Some(cand);
                    }
                }
            }
        }
        None
    }

    /// Check if `ancestor` is an ancestor of `node` (walks up parent chain).
    fn is_ancestor(&self, ancestor: u32, node: u32) -> bool {
        let mut cur = node;
        while cur != NO_PARENT {
            if cur == ancestor {
                return true;
            }
            cur = self.parent[cur as usize];
        }
        false
    }

    /// Collapse unifurcation nodes (inner nodes with exactly one child).
    ///
    /// Iterates until no more unifurcations remain, handling chains correctly.
    pub fn collapse_unifurcations(&mut self) {
        loop {
            let mut did_collapse = false;
            for i in 1..self.len() {
                if self.parent[i] == NO_PARENT {
                    continue;
                }
                if self.is_leaf[i] {
                    continue;
                }
                if self.children[i].len() != 1 {
                    continue;
                }

                let child = self.children[i][0];
                let par = self.parent[i];

                // Reparent the single child to this node's parent
                self.parent[child as usize] = par;

                // Update parent's children: replace i with child
                if par != NO_PARENT {
                    let i32 = i as u32;
                    if let Some(pos) =
                        self.children[par as usize].iter().position(|&c| c == i32)
                    {
                        self.children[par as usize][pos] = child;
                    }
                }

                // Orphan this node
                self.children[i].clear();
                self.parent[i] = NO_PARENT;
                did_collapse = true;
            }
            if !did_collapse {
                break;
            }
        }
    }

    /// Check if a node is reachable from root (node 0).
    pub fn is_reachable(&self, node: u32) -> bool {
        self.is_ancestor(0, node)
    }
}

/// Naive trie for phylogenetic tree reconstruction (legacy fallback).
///
/// Unlike [`Trie`], this does NOT use a search table for descendant lookup.
/// Instead, it walks the tree recursively, making it O(N*D) per insertion
/// where D is the tree depth. Exponential worst case for pathological inputs.
/// Required by the plan for compatibility testing against ShortcutConsolidation.
#[derive(Debug, Clone)]
pub struct NaiveTrie {
    pub parent: Vec<u32>,
    pub rank: Vec<u64>,
    pub differentia: Vec<u64>,
    pub is_leaf: Vec<bool>,
    pub taxon_id: Vec<Option<u32>>,
    pub children: Vec<Vec<u32>>,
}

impl NaiveTrie {
    /// Create a new naive trie with a virtual root node.
    pub fn new() -> Self {
        Self {
            parent: alloc::vec![NO_PARENT],
            rank: alloc::vec![u64::MAX],
            differentia: alloc::vec![0],
            is_leaf: alloc::vec![false],
            taxon_id: alloc::vec![None],
            children: alloc::vec![Vec::new()],
        }
    }

    /// Number of nodes (including virtual root and orphaned nodes).
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Add a node as a child of `parent_idx`.
    pub fn add_node(
        &mut self,
        parent_idx: u32,
        rank: u64,
        diff: u64,
        is_leaf: bool,
        taxon: Option<u32>,
    ) -> u32 {
        let idx = self.parent.len() as u32;
        self.parent.push(parent_idx);
        self.rank.push(rank);
        self.differentia.push(diff);
        self.is_leaf.push(is_leaf);
        self.taxon_id.push(taxon);
        self.children.push(Vec::new());
        self.children[parent_idx as usize].push(idx);
        idx
    }

    /// Find a descendant node with matching `(rank, differentia)` by walking
    /// the subtree rooted at `ancestor`. No search table — pure DFS.
    pub fn find_descendant(
        &self,
        ancestor: u32,
        rank: u64,
        diff: u64,
    ) -> Option<u32> {
        // Direct children first
        for &child in &self.children[ancestor as usize] {
            if !self.is_leaf[child as usize]
                && self.rank[child as usize] == rank
                && self.differentia[child as usize] == diff
            {
                return Some(child);
            }
        }

        // DFS into subtree
        let mut stack: Vec<u32> = self.children[ancestor as usize]
            .iter()
            .copied()
            .collect();
        while let Some(node) = stack.pop() {
            for &child in &self.children[node as usize] {
                if !self.is_leaf[child as usize]
                    && self.rank[child as usize] == rank
                    && self.differentia[child as usize] == diff
                {
                    return Some(child);
                }
                stack.push(child);
            }
        }
        None
    }

    /// Collapse unifurcation nodes (inner nodes with exactly one child).
    pub fn collapse_unifurcations(&mut self) {
        loop {
            let mut did_collapse = false;
            for i in 1..self.len() {
                if self.parent[i] == NO_PARENT {
                    continue;
                }
                if self.is_leaf[i] {
                    continue;
                }
                if self.children[i].len() != 1 {
                    continue;
                }

                let child = self.children[i][0];
                let par = self.parent[i];

                self.parent[child as usize] = par;

                if par != NO_PARENT {
                    let i32 = i as u32;
                    if let Some(pos) =
                        self.children[par as usize].iter().position(|&c| c == i32)
                    {
                        self.children[par as usize][pos] = child;
                    }
                }

                self.children[i].clear();
                self.parent[i] = NO_PARENT;
                did_collapse = true;
            }
            if !did_collapse {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_empty_trie() {
        let trie = Trie::new();
        assert_eq!(trie.len(), 1);
        assert_eq!(trie.rank[0], u64::MAX); // virtual root
    }

    #[test]
    fn add_inner_and_leaf() {
        let mut trie = Trie::new();
        let inner = trie.add_node(0, 0, 42, false, None);
        assert_eq!(inner, 1);
        assert!(!trie.is_leaf[1]);

        let leaf = trie.add_node(inner, 1, 99, true, Some(0));
        assert_eq!(leaf, 2);
        assert!(trie.is_leaf[2]);
        assert_eq!(trie.taxon_id[2], Some(0));
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn find_descendant_direct_child() {
        let mut trie = Trie::new();
        let inner = trie.add_node(0, 0, 42, false, None);
        let found = trie.find_descendant(0, 0, 42);
        assert_eq!(found, Some(inner));
    }

    #[test]
    fn find_descendant_grandchild() {
        let mut trie = Trie::new();
        let a = trie.add_node(0, 0, 10, false, None);
        let b = trie.add_node(a, 5, 20, false, None);
        let _c = trie.add_node(b, 10, 30, false, None);

        // Find grandchild of root at rank 10
        let found = trie.find_descendant(0, 10, 30);
        assert_eq!(found, Some(3)); // node c
    }

    #[test]
    fn find_descendant_not_found() {
        let mut trie = Trie::new();
        trie.add_node(0, 0, 42, false, None);
        assert_eq!(trie.find_descendant(0, 0, 99), None);
        assert_eq!(trie.find_descendant(0, 5, 42), None);
    }

    #[test]
    fn find_descendant_skips_leaves() {
        let mut trie = Trie::new();
        trie.add_node(0, 0, 42, true, Some(0)); // leaf
        // Should not find the leaf
        assert_eq!(trie.find_descendant(0, 0, 42), None);
    }

    #[test]
    fn collapse_simple_chain() {
        let mut trie = Trie::new();
        let a = trie.add_node(0, 0, 10, false, None);
        let b = trie.add_node(a, 1, 20, false, None);
        let c = trie.add_node(b, 2, 30, false, None);
        // Branch at c
        trie.add_node(c, 3, 40, true, Some(0));
        trie.add_node(c, 3, 50, true, Some(1));

        trie.collapse_unifurcations();

        // a and b should be collapsed; c has 2 children so stays
        assert_eq!(trie.parent[c as usize], 0); // c now child of root
        assert_eq!(trie.parent[a as usize], NO_PARENT); // a orphaned
        assert_eq!(trie.parent[b as usize], NO_PARENT); // b orphaned
    }

    #[test]
    fn collapse_preserves_branches() {
        let mut trie = Trie::new();
        let a = trie.add_node(0, 0, 10, false, None);
        trie.add_node(a, 1, 20, true, Some(0));
        trie.add_node(a, 1, 30, true, Some(1));

        trie.collapse_unifurcations();

        // a has 2 children → not collapsed
        assert_eq!(trie.parent[a as usize], 0);
        assert_eq!(trie.children[a as usize].len(), 2);
    }
}

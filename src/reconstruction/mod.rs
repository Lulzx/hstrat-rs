pub mod build_tree;
pub mod mrca;
pub mod trie;

pub use build_tree::{build_tree, AlifeDataFrame, TreeAlgorithm};
pub use mrca::{
    calc_rank_of_mrca_bounds_between, calc_ranks_since_mrca_bounds_between,
    does_have_any_common_ancestor,
};

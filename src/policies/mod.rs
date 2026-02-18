pub mod curbed_recency_proportional;
pub mod depth_proportional;
pub mod depth_proportional_tapered;
pub mod dynamic;
pub mod fixed_resolution;
pub mod geometric_seq_nth_root;
pub mod geometric_seq_nth_root_tapered;
pub mod nominal_resolution;
pub mod perfect_resolution;
pub mod pseudostochastic;
pub mod recency_proportional;
pub mod stochastic;
mod r#trait;

/// Compute `lhs \ rhs` for sorted ascending unique rank slices.
pub(crate) fn sorted_set_difference(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            core::cmp::Ordering::Less => {
                out.push(lhs[i]);
                i += 1;
            }
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Greater => {
                j += 1;
            }
        }
    }

    out.extend_from_slice(&lhs[i..]);
    out
}

/// Return the first element in `lhs` that is not present in sorted `rhs`.
pub(crate) fn first_sorted_set_difference(lhs: &[u64], rhs: &[u64]) -> Option<u64> {
    let mut i = 0usize;
    let mut j = 0usize;

    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            core::cmp::Ordering::Less => return Some(lhs[i]),
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Greater => {
                j += 1;
            }
        }
    }

    lhs.get(i).copied()
}

pub use curbed_recency_proportional::CurbedRecencyProportionalPolicy;
pub use depth_proportional::DepthProportionalPolicy;
pub use depth_proportional_tapered::DepthProportionalTaperedPolicy;
pub use dynamic::DynamicPolicy;
pub use fixed_resolution::FixedResolutionPolicy;
pub use geometric_seq_nth_root::GeometricSeqNthRootPolicy;
pub use geometric_seq_nth_root_tapered::GeometricSeqNthRootTaperedPolicy;
pub use nominal_resolution::NominalResolutionPolicy;
pub use perfect_resolution::PerfectResolutionPolicy;
pub use pseudostochastic::PseudostochasticPolicy;
pub use r#trait::StratumRetentionPolicy;
pub use recency_proportional::RecencyProportionalPolicy;
pub use stochastic::StochasticPolicy;

#[cfg(test)]
mod tests {
    use super::{first_sorted_set_difference, sorted_set_difference};

    #[test]
    fn sorted_set_difference_basic() {
        let lhs = [0u64, 1, 2, 4, 8];
        let rhs = [1u64, 3, 4, 9];
        assert_eq!(sorted_set_difference(&lhs, &rhs), vec![0, 2, 8]);
    }

    #[test]
    fn first_sorted_set_difference_basic() {
        let lhs = [2u64, 4, 6, 8];
        let rhs = [1u64, 2, 5, 8];
        assert_eq!(first_sorted_set_difference(&lhs, &rhs), Some(4));
        assert_eq!(first_sorted_set_difference(&rhs, &rhs), None);
    }
}

use alloc::string::String;
use core::fmt;

/// Error types for the hstrat crate.
#[derive(Debug, Clone, PartialEq)]
pub enum HstratError {
    /// Differentia bit width must be in 1..=64.
    InvalidBitWidth(u8),
    /// A policy parameter is invalid.
    InvalidPolicyParam { param: &'static str, value: i64 },
    /// Deserialization failed.
    DeserializationError(String),
    /// Two columns cannot be compared (incompatible).
    IncompatibleColumns,
    /// No common ancestor detected between two columns.
    NoCommonAncestor,
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, len: usize },
}

impl fmt::Display for HstratError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBitWidth(w) => {
                write!(f, "invalid differentia bit width: {} (must be 1..=64)", w)
            }
            Self::InvalidPolicyParam { param, value } => {
                write!(f, "invalid policy parameter '{}': {}", param, value)
            }
            Self::DeserializationError(msg) => {
                write!(f, "deserialization error: {}", msg)
            }
            Self::IncompatibleColumns => {
                write!(f, "columns are incompatible for comparison")
            }
            Self::NoCommonAncestor => {
                write!(f, "no common ancestor detected")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {} out of bounds for length {}", index, len)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HstratError {}

pub type Result<T> = core::result::Result<T, HstratError>;

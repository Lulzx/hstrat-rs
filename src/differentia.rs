/// A differentia value — the random fingerprint assigned to each stratum.
///
/// Internally stored as a `u64`, but only the lower `bit_width` bits are
/// meaningful. All comparisons and hashing use the full stored value, which
/// is always pre-masked on construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Differentia(u64);

impl Differentia {
    /// Create a new differentia, masking to `bit_width` lower bits.
    ///
    /// # Panics (debug only)
    /// Panics if `bit_width` is 0 or greater than 64.
    #[inline]
    pub fn new(value: u64, bit_width: u8) -> Self {
        debug_assert!(
            bit_width >= 1 && bit_width <= 64,
            "bit_width must be in 1..=64, got {}",
            bit_width
        );
        Self(value & Self::mask(bit_width))
    }

    /// Bit mask for the given width: the lower `bit_width` bits set.
    #[inline]
    pub fn mask(bit_width: u8) -> u64 {
        if bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        }
    }

    /// Check if two differentia match at the given bit width.
    #[inline]
    pub fn matches(self, other: Self, bit_width: u8) -> bool {
        let m = Self::mask(bit_width);
        (self.0 & m) == (other.0 & m)
    }

    /// Raw u64 value (already masked to bit_width on construction).
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Differentia {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Differentia {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = u64::deserialize(deserializer)?;
        Ok(Self(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_boundaries() {
        assert_eq!(Differentia::mask(1), 1);
        assert_eq!(Differentia::mask(8), 0xFF);
        assert_eq!(Differentia::mask(32), 0xFFFF_FFFF);
        assert_eq!(Differentia::mask(63), (1u64 << 63) - 1);
        assert_eq!(Differentia::mask(64), u64::MAX);
    }

    #[test]
    fn new_masks_correctly() {
        let d = Differentia::new(0xDEAD_BEEF_CAFE_BABE, 8);
        assert_eq!(d.value(), 0xBE);

        let d = Differentia::new(0xDEAD_BEEF_CAFE_BABE, 1);
        assert_eq!(d.value(), 0);

        let d = Differentia::new(u64::MAX, 64);
        assert_eq!(d.value(), u64::MAX);

        let d = Differentia::new(0xFF, 4);
        assert_eq!(d.value(), 0xF);
    }

    #[test]
    fn matches_with_masking() {
        let a = Differentia::new(0b1010_1111, 4);
        let b = Differentia::new(0b0000_1111, 4);
        assert!(a.matches(b, 4));

        let a = Differentia::new(0b1010_1110, 4);
        let b = Differentia::new(0b0000_1111, 4);
        assert!(!a.matches(b, 4));
    }

    #[test]
    fn matches_full_width() {
        let a = Differentia::new(42, 64);
        let b = Differentia::new(42, 64);
        assert!(a.matches(b, 64));

        let c = Differentia::new(43, 64);
        assert!(!a.matches(c, 64));
    }
}

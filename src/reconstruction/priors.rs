/// Trait for MRCA prior distributions used in probabilistic estimation.
///
/// A prior encodes belief about where the MRCA might fall within an interval
/// `[begin, end)` (in terms of ranks). Two methods are required:
/// - `calc_interval_probability_proxy` — unnormalized weight for the interval
/// - `calc_interval_conditioned_mean` — expected MRCA rank within the interval
pub trait Prior: Send + Sync {
    /// Return an unnormalized weight proportional to the probability that
    /// the MRCA falls in `[begin, end)`.
    fn calc_interval_probability_proxy(&self, begin: u64, end: u64) -> f64;

    /// Return the expected MRCA rank conditioned on it falling in `[begin, end)`.
    fn calc_interval_conditioned_mean(&self, begin: u64, end: u64) -> f64;
}

// ---------------------------------------------------------------------------
// ArbitraryPrior
// ---------------------------------------------------------------------------

/// A non-informative prior that assigns equal weight to all intervals.
///
/// Useful when no information about generation time or growth rate is known.
/// `prob_proxy` always returns 1.0; `conditioned_mean` returns the midpoint.
pub struct ArbitraryPrior;

impl Prior for ArbitraryPrior {
    fn calc_interval_probability_proxy(&self, _begin: u64, _end: u64) -> f64 {
        1.0
    }

    fn calc_interval_conditioned_mean(&self, begin: u64, end: u64) -> f64 {
        (begin as f64 + end as f64 - 1.0) / 2.0
    }
}

// ---------------------------------------------------------------------------
// UniformPrior
// ---------------------------------------------------------------------------

/// A uniform prior that weights each rank equally (probability ∝ interval width).
///
/// `prob_proxy` returns `(end - begin)` as a weight proportional to interval length.
pub struct UniformPrior;

impl Prior for UniformPrior {
    fn calc_interval_probability_proxy(&self, begin: u64, end: u64) -> f64 {
        (end - begin) as f64
    }

    fn calc_interval_conditioned_mean(&self, begin: u64, end: u64) -> f64 {
        (begin as f64 + end as f64 - 1.0) / 2.0
    }
}

// ---------------------------------------------------------------------------
// ExponentialPrior
// ---------------------------------------------------------------------------

/// An exponential growth prior where population size grows as `growth_factor^rank`.
///
/// More recent ranks are weighted more heavily when `growth_factor > 1.0`.
///
/// `prob_proxy(begin, end) = growth_factor^end - growth_factor^begin`
///
/// Special case: `growth_factor == 1.0` degenerates to `UniformPrior`.
pub struct ExponentialPrior {
    pub growth_factor: f64,
}

impl ExponentialPrior {
    pub fn new(growth_factor: f64) -> Self {
        assert!(
            growth_factor > 0.0,
            "growth_factor must be positive, got {}",
            growth_factor
        );
        Self { growth_factor }
    }
}

impl Prior for ExponentialPrior {
    fn calc_interval_probability_proxy(&self, begin: u64, end: u64) -> f64 {
        let g = self.growth_factor;
        if (g - 1.0).abs() < f64::EPSILON {
            return (end - begin) as f64;
        }
        libm::pow(g, end as f64) - libm::pow(g, begin as f64)
    }

    fn calc_interval_conditioned_mean(&self, begin: u64, end: u64) -> f64 {
        let g = self.growth_factor;
        // Degenerate case: uniform distribution
        if (g - 1.0).abs() < f64::EPSILON {
            return (begin as f64 + end as f64 - 1.0) / 2.0;
        }

        // Weighted mean: ∫(begin..end) x * g^x dx / ∫(begin..end) g^x dx
        //
        // ∫ g^x dx = g^x / ln(g)
        // ∫ x*g^x dx = g^x*(x*ln(g) - 1) / ln(g)^2
        //
        // Numerator = [g^x*(x*ln(g) - 1)/ln(g)^2] from begin to end
        // Denominator = [g^x/ln(g)] from begin to end = (g^end - g^begin)/ln(g)
        let b = begin as f64;
        let e = end as f64;
        let ln_g = libm::log(g);

        let g_b = libm::pow(g, b);
        let g_e = libm::pow(g, e);

        let numerator = (g_e * (e * ln_g - 1.0) - g_b * (b * ln_g - 1.0)) / (ln_g * ln_g);
        let denominator = (g_e - g_b) / ln_g;

        if denominator.abs() < f64::EPSILON {
            // Numerically degenerate; fall back to midpoint
            (b + e - 1.0) / 2.0
        } else {
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_prior_proxy_is_one() {
        let p = ArbitraryPrior;
        assert_eq!(p.calc_interval_probability_proxy(0, 100), 1.0);
        assert_eq!(p.calc_interval_probability_proxy(50, 51), 1.0);
    }

    #[test]
    fn arbitrary_prior_mean_is_midpoint() {
        let p = ArbitraryPrior;
        // [0, 10) → midpoint = (0 + 10 - 1) / 2 = 4.5
        assert!((p.calc_interval_conditioned_mean(0, 10) - 4.5).abs() < 1e-10);
        // [5, 6) → midpoint = (5 + 6 - 1) / 2 = 5.0
        assert!((p.calc_interval_conditioned_mean(5, 6) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn uniform_prior_proxy_is_width() {
        let p = UniformPrior;
        assert_eq!(p.calc_interval_probability_proxy(0, 10), 10.0);
        assert_eq!(p.calc_interval_probability_proxy(3, 7), 4.0);
    }

    #[test]
    fn exponential_prior_growth_factor_1_is_uniform() {
        let exp = ExponentialPrior::new(1.0);
        let uni = UniformPrior;
        for (b, e) in [(0u64, 10u64), (5, 20), (100, 200)] {
            let ep = exp.calc_interval_probability_proxy(b, e);
            let up = uni.calc_interval_probability_proxy(b, e);
            assert!((ep - up).abs() < 1e-8, "proxy mismatch at ({b},{e}): {ep} vs {up}");
            let em = exp.calc_interval_conditioned_mean(b, e);
            let um = uni.calc_interval_conditioned_mean(b, e);
            assert!((em - um).abs() < 1e-8, "mean mismatch at ({b},{e}): {em} vs {um}");
        }
    }

    #[test]
    fn exponential_prior_mean_in_interval() {
        let exp = ExponentialPrior::new(1.1);
        let mean = exp.calc_interval_conditioned_mean(0, 100);
        assert!(mean >= 0.0 && mean < 100.0);
    }

    #[test]
    fn exponential_prior_proxy_positive() {
        let exp = ExponentialPrior::new(2.0);
        assert!(exp.calc_interval_probability_proxy(0, 10) > 0.0);
        assert!(exp.calc_interval_probability_proxy(5, 6) > 0.0);
    }

    #[test]
    fn exponential_prior_mean_biased_toward_end() {
        // With growth_factor > 1, more weight at higher ranks → mean > midpoint
        let exp = ExponentialPrior::new(2.0);
        let mean = exp.calc_interval_conditioned_mean(0, 100);
        let midpoint = 49.5;
        assert!(mean > midpoint, "mean={mean} should be > midpoint={midpoint}");
    }
}

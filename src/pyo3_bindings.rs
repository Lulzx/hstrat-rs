// PyO3 bindings — behind `pyo3` feature flag.
// Exposes HereditaryStratigraphicColumn with all 11 policies via DynamicPolicy
// runtime dispatch, plus build_tree and MRCA functions.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
use crate::column::HereditaryStratigraphicColumn;
#[cfg(feature = "pyo3")]
use crate::policies::{DynamicPolicy, StratumRetentionPolicy};

#[cfg(feature = "pyo3")]
#[pyclass(name = "HereditaryStratigraphicColumn")]
#[derive(Clone)]
pub struct PyHereditaryStratigraphicColumn {
    inner: HereditaryStratigraphicColumn<DynamicPolicy>,
}

#[cfg(feature = "pyo3")]
impl PyHereditaryStratigraphicColumn {
    fn get_kwarg_u64(kwargs: &Option<Bound<'_, PyDict>>, key: &str, default: u64) -> PyResult<u64> {
        if let Some(kw) = kwargs {
            if let Some(v) = kw.get_item(key)? {
                return v.extract::<u64>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "keyword '{key}' must be a non-negative integer"
                    ))
                });
            }
        }
        Ok(default)
    }

    fn get_kwarg_f64(kwargs: &Option<Bound<'_, PyDict>>, key: &str, default: f64) -> PyResult<f64> {
        if let Some(kw) = kwargs {
            if let Some(v) = kw.get_item(key)? {
                return v.extract::<f64>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "keyword '{key}' must be a number"
                    ))
                });
            }
        }
        Ok(default)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyHereditaryStratigraphicColumn {
    #[new]
    #[pyo3(signature = (policy_name, differentia_bit_width=64, **kwargs))]
    fn new(
        policy_name: &str,
        differentia_bit_width: u8,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        use crate::policies::*;

        if !(1..=64).contains(&differentia_bit_width) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "differentia_bit_width must be in 1..=64, got {}",
                differentia_bit_width
            )));
        }

        let policy = match policy_name {
            "fixed_resolution" => {
                let resolution = Self::get_kwarg_u64(&kwargs, "resolution", 10)?;
                if resolution == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "resolution must be positive",
                    ));
                }
                DynamicPolicy::FixedResolution(FixedResolutionPolicy::new(resolution))
            }
            "perfect_resolution" => DynamicPolicy::PerfectResolution(PerfectResolutionPolicy),
            "nominal_resolution" => DynamicPolicy::NominalResolution(NominalResolutionPolicy),
            "recency_proportional" => {
                let resolution = Self::get_kwarg_u64(&kwargs, "resolution", 3)?;
                if resolution == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "resolution must be positive",
                    ));
                }
                DynamicPolicy::RecencyProportional(RecencyProportionalPolicy::new(resolution))
            }
            "curbed_recency_proportional" => {
                let size_curb = Self::get_kwarg_u64(&kwargs, "size_curb", 10)?;
                if size_curb < 8 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "size_curb must be at least 8",
                    ));
                }
                DynamicPolicy::CurbedRecencyProportional(CurbedRecencyProportionalPolicy::new(
                    size_curb,
                ))
            }
            "geometric_seq_nth_root" => {
                let degree = Self::get_kwarg_u64(&kwargs, "degree", 2)?;
                let interspersal = Self::get_kwarg_u64(&kwargs, "interspersal", 2)?;
                if degree == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "degree must be positive",
                    ));
                }
                if interspersal == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "interspersal must be positive",
                    ));
                }
                DynamicPolicy::GeometricSeqNthRoot(GeometricSeqNthRootPolicy::new(
                    degree,
                    interspersal,
                ))
            }
            "geometric_seq_nth_root_tapered" => {
                let degree = Self::get_kwarg_u64(&kwargs, "degree", 2)?;
                let interspersal = Self::get_kwarg_u64(&kwargs, "interspersal", 2)?;
                if degree == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "degree must be positive",
                    ));
                }
                if interspersal == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "interspersal must be positive",
                    ));
                }
                DynamicPolicy::GeometricSeqNthRootTapered(GeometricSeqNthRootTaperedPolicy::new(
                    degree,
                    interspersal,
                ))
            }
            "depth_proportional" => {
                let resolution = Self::get_kwarg_u64(&kwargs, "resolution", 5)?;
                if resolution == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "resolution must be positive",
                    ));
                }
                DynamicPolicy::DepthProportional(DepthProportionalPolicy::new(resolution))
            }
            "depth_proportional_tapered" => {
                let resolution = Self::get_kwarg_u64(&kwargs, "resolution", 5)?;
                if resolution == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "resolution must be positive",
                    ));
                }
                DynamicPolicy::DepthProportionalTapered(DepthProportionalTaperedPolicy::new(
                    resolution,
                ))
            }
            "pseudostochastic" => {
                let hash_salt = Self::get_kwarg_u64(&kwargs, "hash_salt", 0)?;
                DynamicPolicy::Pseudostochastic(PseudostochasticPolicy::new(hash_salt))
            }
            "stochastic" => {
                let prob = Self::get_kwarg_f64(&kwargs, "retention_probability", 0.5)?;
                if !prob.is_finite() || !(0.0..=1.0).contains(&prob) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "retention_probability must be in 0.0..=1.0",
                    ));
                }
                DynamicPolicy::Stochastic(StochasticPolicy::new(prob))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown policy: {}",
                    policy_name
                )));
            }
        };

        // Use system time for seed (pyo3 implies std)
        let seed = {
            use std::time::SystemTime;
            let dur = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default();
            dur.as_nanos() as u64
        };

        Ok(Self {
            inner: HereditaryStratigraphicColumn::with_seed(policy, differentia_bit_width, seed),
        })
    }

    fn deposit_stratum(&mut self) {
        self.inner.deposit_stratum();
    }

    fn deposit_strata(&mut self, n: u64) {
        self.inner.deposit_strata(n);
    }

    fn clone_descendant(&self) -> Self {
        Self {
            inner: self.inner.clone_descendant(),
        }
    }

    fn get_num_strata_deposited(&self) -> u64 {
        self.inner.get_num_strata_deposited()
    }

    fn get_num_strata_retained(&self) -> usize {
        self.inner.get_num_strata_retained()
    }

    fn get_stratum_differentia_bit_width(&self) -> u8 {
        self.inner.get_stratum_differentia_bit_width()
    }

    fn get_retained_ranks(&self) -> Vec<u64> {
        self.inner.iter_retained_ranks().collect()
    }

    fn get_retained_differentia(&self) -> Vec<u64> {
        self.inner
            .iter_retained_differentia()
            .map(|d| d.value())
            .collect()
    }

    fn get_policy_algo_identifier(&self) -> &'static str {
        self.inner.get_policy().algo_identifier()
    }
}

/// Check whether two columns share any common ancestor.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn does_have_any_common_ancestor(
    a: &PyHereditaryStratigraphicColumn,
    b: &PyHereditaryStratigraphicColumn,
) -> bool {
    crate::reconstruction::does_have_any_common_ancestor(&a.inner, &b.inner)
}

/// Calculate MRCA rank bounds between two columns.
/// Returns (lower_bound, upper_bound) or None.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn calc_rank_of_mrca_bounds_between(
    a: &PyHereditaryStratigraphicColumn,
    b: &PyHereditaryStratigraphicColumn,
) -> Option<(u64, u64)> {
    crate::reconstruction::calc_rank_of_mrca_bounds_between(&a.inner, &b.inner)
}

/// Build a phylogenetic tree from a population of columns.
/// Returns a dict with keys: id, ancestor_list, origin_time, taxon_label.
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (population, algorithm="shortcut", taxon_labels=None))]
fn build_tree(
    py: Python<'_>,
    population: Vec<PyHereditaryStratigraphicColumn>,
    algorithm: &str,
    taxon_labels: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    use crate::reconstruction::TreeAlgorithm;

    let algo = match algorithm {
        "shortcut" | "shortcut_consolidation" => TreeAlgorithm::ShortcutConsolidation,
        "naive" | "naive_trie" => TreeAlgorithm::NaiveTrie,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown algorithm: {}",
                algorithm
            )))
        }
    };

    let cols: Vec<_> = population.iter().map(|p| p.inner.clone()).collect();
    let labels_ref = taxon_labels.as_deref();
    let df = crate::reconstruction::build_tree(&cols, algo, labels_ref, None);

    let dict = PyDict::new(py);
    dict.set_item("id", df.id)?;
    dict.set_item("ancestor_list", df.ancestor_list)?;
    dict.set_item("origin_time", df.origin_time)?;
    dict.set_item("taxon_label", df.taxon_label)?;
    Ok(dict.into())
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn hstrat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHereditaryStratigraphicColumn>()?;
    m.add_function(wrap_pyfunction!(does_have_any_common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(calc_rank_of_mrca_bounds_between, m)?)?;
    m.add_function(wrap_pyfunction!(build_tree, m)?)?;
    Ok(())
}

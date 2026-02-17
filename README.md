# hstrat

Rust implementation of hereditary stratigraphy for phylogenetic
inference. Port of the Python
[hstrat](https://github.com/mmore500/hstrat) library.

Hereditary stratigraphy enables decentralized phylogenetic tracking in
large-scale digital evolution experiments. Each organism carries a
compact column of retained checkpoints (strata); comparing columns
between organisms reconstructs their evolutionary relationships
without centralized bookkeeping.

## Features

- 11 stratum retention policies (fixed resolution, depth proportional,
  geometric, recency proportional, tapered variants, etc.)
- `DynamicPolicy` enum for runtime policy dispatch
- Pairwise MRCA (most recent common ancestor) bound estimation
- Trie-based tree reconstruction with search-table acceleration
  (two algorithms: shortcut-consolidation + naive DFS fallback)
- Binary packet and JSON record serialization, round-trip compatible
  with Python hstrat
- `no_std` compatible core (`alloc` only); optional `std`, `serde`,
  `pyo3`, `rayon` features
- Python bindings via PyO3 (all policies, MRCA, tree building)
- Cross-validated against Python hstrat: 22 policy configurations
  verified for n=0..1000, plus MRCA bounds and serialization packets

## Performance

Criterion benchmarks on Apple M-series:

| Operation                    | Time     |
|------------------------------|----------|
| `build_tree` 10k tips        | ~3.5 ms  |
| `deposit` 1M (perfect res.)  | ~3.2 ms  |
| `deposit` 1M (fixed res. 10) | ~4.6 ms  |
| MRCA 10k pairs               | ~642 µs  |

## Usage

```rust
use hstrat::column::HereditaryStratigraphicColumn;
use hstrat::policies::FixedResolutionPolicy;
use hstrat::reconstruction::{build_tree, TreeAlgorithm};

// create a column with fixed-resolution retention (keep every 10th rank)
let mut ancestor = HereditaryStratigraphicColumn::with_seed(
    FixedResolutionPolicy::new(10), 64, 42,
);
ancestor.deposit_strata(100);

// simulate a population
let mut population = Vec::new();
for _ in 0..50 {
    let mut child = ancestor.clone_descendant();
    child.deposit_strata(20);
    population.push(child);
}

// reconstruct the phylogenetic tree
let tree = build_tree(&population, TreeAlgorithm::ShortcutConsolidation, None);
println!("{} nodes in reconstructed tree", tree.len());
```

## Building

```sh
cargo build
cargo test
cargo bench
```

Python extension (requires [maturin](https://github.com/PyO3/maturin)):

```sh
maturin develop --features pyo3
```

## Cross-validation

Test vectors are extracted from Python hstrat and stored in
`tests/fixtures/`. To regenerate:

```sh
python -m venv .venv
.venv/bin/pip install hstrat
.venv/bin/python scripts/extract_test_vectors.py
```

226 tests (178 unit + 48 integration) verify internal consistency
and bit-for-bit compatibility with Python hstrat.

## License

MIT

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hstrat::column::HereditaryStratigraphicColumn;
use hstrat::policies::*;
use hstrat::reconstruction;

fn bench_deposit_strata_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("deposit_strata_1m");
    group.sample_size(10);

    group.bench_function("perfect_resolution", |b| {
        b.iter(|| {
            let mut col = HereditaryStratigraphicColumn::with_seed(
                PerfectResolutionPolicy,
                64,
                42,
            );
            col.deposit_strata(black_box(1_000_000));
        });
    });

    group.bench_function("fixed_resolution_10", |b| {
        b.iter(|| {
            let mut col = HereditaryStratigraphicColumn::with_seed(
                FixedResolutionPolicy::new(10),
                64,
                42,
            );
            col.deposit_strata(black_box(1_000_000));
        });
    });

    group.bench_function("nominal_resolution", |b| {
        b.iter(|| {
            let mut col = HereditaryStratigraphicColumn::with_seed(
                NominalResolutionPolicy,
                64,
                42,
            );
            col.deposit_strata(black_box(1_000_000));
        });
    });

    group.finish();
}

fn bench_mrca_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("mrca_pairwise");
    group.sample_size(10);

    group.bench_function("10k_pairs_fixed_res", |b| {
        let mut ancestor = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(10),
            64,
            42,
        );
        ancestor.deposit_strata(100);

        let mut children = Vec::with_capacity(100);
        for i in 0..100u64 {
            let mut child = ancestor.clone_descendant();
            child.deposit_strata(i);
            children.push(child);
        }

        b.iter(|| {
            let mut count = 0u64;
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    if let Some((lo, _hi)) =
                        reconstruction::calc_rank_of_mrca_bounds_between(
                            black_box(&children[i]),
                            black_box(&children[j]),
                        )
                    {
                        count += lo;
                    }
                }
            }
            count
        });
    });

    group.finish();
}

fn bench_build_tree_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_tree");
    group.sample_size(10);

    group.bench_function("10k_tips_fixed_res", |b| {
        let mut ancestor = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(10),
            64,
            42,
        );
        ancestor.deposit_strata(100);

        let mut population = Vec::with_capacity(10_000);
        for i in 0..10_000u64 {
            let mut child = ancestor.clone_descendant();
            child.deposit_strata(i % 50);
            population.push(child);
        }

        b.iter(|| {
            reconstruction::build_tree(
                black_box(&population),
                reconstruction::TreeAlgorithm::ShortcutConsolidation,
                None,
            );
        });
    });

    group.finish();
}

fn bench_build_tree_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_tree");
    group.sample_size(10);

    group.bench_function("100k_tips_fixed_res", |b| {
        let mut ancestor = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(10),
            64,
            42,
        );
        ancestor.deposit_strata(100);

        let mut population = Vec::with_capacity(100_000);
        for i in 0..100_000u64 {
            let mut child = ancestor.clone_descendant();
            child.deposit_strata(i % 50);
            population.push(child);
        }

        b.iter(|| {
            reconstruction::build_tree(
                black_box(&population),
                reconstruction::TreeAlgorithm::ShortcutConsolidation,
                None,
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_deposit_strata_1m,
    bench_mrca_10k,
    bench_build_tree_10k,
    bench_build_tree_100k
);
criterion_main!(benches);

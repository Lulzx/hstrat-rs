# Code Quality Rules

## Optimizations must be opt-in, not opt-out

Never generalize a fast path observed in one case to all cases. Fast paths must be gated on an explicit invariant (a trait method, a type-level guarantee, or a proven property) rather than assumed from the common case. The default code path must always be correct; the fast path is added only after the invariant is verified and encoded.

## Fail loudly at system boundaries

At every system boundary — constructors, deserialization, public API entry points — validate inputs and return errors or panic on invalid state. Never silently substitute a default (`unwrap_or(0)`, fallback values) for missing or malformed data. Silent substitution turns bugs into silent corruption that is much harder to diagnose than an early, explicit failure.

## Never assume homogeneity across two instances of the same type

When a function takes two values of the same type, derive any shared property from both, not just one. Bit widths, resolutions, policies, and other per-instance configuration may differ. Always compute the shared quantity explicitly (e.g., `min(a.width, b.width)`) and document why.

## `debug_assert!` is not a contract

Use `assert!` for invariants that must hold in production. Use `debug_assert!` only for expensive checks that are purely defensive and where silent violation in release mode is acceptable. Invariants at constructor and deserialization boundaries are never acceptable to skip in release mode — they belong in `assert!`.

## Test the assumption-violating case, not just the common case

For every design assumption, add a test that violates it. If a fast path assumes property P, write a test with a case where P does not hold and verify the slow path handles it correctly. Tests that only cover the common case provide false confidence.

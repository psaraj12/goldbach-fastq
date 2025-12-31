# Changelog

All notable changes to this project are documented here.
This project follows semantic versioning.


v2.0 — 2025-12-31
Added

Deterministic, full-coverage Goldbach verification engine

Winner-biased FastQ for adaptive prime reuse

Elastic FastQ scan windows (32 → 128 → 256)

Bounded small-prime sieve stage for early resolution

Δd walker fallback around N/2

Emergency fallback stage (never triggered in tested ranges)

OpenMP parallel execution with per-thread timing

Changed

Execution model redesigned from heuristic search to staged deterministic pipeline

Prime candidate ordering biased toward recent successful offsets

Small-prime handling separated from large-prime MR checks

Timing model updated to use per-thread accumulation

Fixed

Duplicate FastQ entries under parallel execution

Non-deterministic ordering effects across threads

Coverage accounting inconsistencies at large ranges

Performance

Verified continuous ranges up to 10¹⁸

Typical coverage achieved without Δd fallback

Significant reduction in cycles per N compared to v1.x
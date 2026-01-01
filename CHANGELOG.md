# Changelog

All notable changes to this project are documented here.
This project follows semantic versioning.


v2.0 — 2025-12-31
Added

## v2.0
- Tuned SEEN mask to reduce FastQ duplicate pollution
- Improved scan stability under heavy prime reuse
- Verified full coverage over 5e9 evens
- Sustained ~1332.6 cycles/N (total-core)


## v2.0.1 – Continuity Stability Patch

### Fixes
- Preserved last-winner continuity across consecutive N
- Prevented side-flipping of reused primes
- Improved FastQ learning stability

### Performance
- ~4.5k cycles / even (total-core) at N ≈ 1e18 (24 threads)
- ~38 seconds for 5×10(power(8)) evens

No algorithmic changes. No correctness trade-offs.
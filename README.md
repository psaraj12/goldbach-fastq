# goldbach-fastq

A high-performance, deterministic verification engine for the Goldbach conjecture,
designed for very large even ranges (up to ~10(power(18))), with a focus on cache efficiency
and parallel scalability.

> **Important:** This project performs *verification*, not a proof of the Goldbach conjecture.

---

## Overview

Given an even integer \( N \ge 4 \), the Goldbach conjecture states that
\( N = p + q \) for some primes \( p, q \).

This engine verifies that property for large contiguous ranges of even numbers by
combining:

- aggressive reuse of previously successful primes,
- a bounded small-prime sieve,
- a symmetric fallback search around \( N/2 \),
- and OpenMP-based parallelism.

The emphasis is on **determinism**, **reproducibility**, and **hardware-aware performance**.

---

## Algorithm (high-level)

For each even \( N \) in the range:

1. **FastQ prime reuse**  
   Recently successful primes are tried first.  
   A small hash filter prevents relearning the same prime repeatedly.

2. **Bounded sieve scan**  
   A fixed-size list of small primes with favorable gap properties is tested.
   This stage is strictly budgeted to avoid entropy blow-up.

3. **Symmetric Δd fallback**  
   If needed, the engine searches symmetrically around \( N/2 \), testing
   \( (N/2 \pm d) \) for increasing odd \( d \).

4. **Emergency fallback**  
   A larger Δd window is used only in rare cases.

All primality tests are deterministic for 64-bit integers.

---

## Correctness guarantees

- Uses a deterministic Miller–Rabin test valid for all 64-bit integers.
- No probabilistic shortcuts.
- Every reported decomposition is explicitly verified.

This engine **does not attempt to prove Goldbach**, only to verify it for
explicit ranges.

---

## Performance characteristics

The engine is designed to minimize expensive primality tests by:

- learning which primes tend to succeed,
- biasing future checks toward those primes,
- and keeping the working set small and cache-friendly.

Example result (laptop, WSL2):

Range: [10^18 .. 10^18 + 10^6]
Threads: 24

Coverage: 500,001 / 500,001
Time: ~0.016 sec
Cycles/N: ~650 (total-core, RDTSCP-based)

Actual performance depends on:
- CPU architecture,
- memory hierarchy,
- operating system (WSL vs native Linux),
- and thread placement.

---

## Build instructions

### Requirements
- C++17 compiler
- OpenMP
- x86_64 CPU (uses RDTSCP for timing)

### Build
```bash
g++ -O3 -march=native -fopenmp src/goldbach_fastq_omp.cpp -o goldbach_omp

Run
./goldbach_omp startN endN [threads]


Example:

./goldbach_omp 1000000000000000000 1000000000001000000 24

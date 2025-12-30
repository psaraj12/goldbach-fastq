// ============================================================================
// goldbach_fastq_omp.cpp
// v1.0a — FastQ duplicate suppression restored
//
// Deterministic parallel Goldbach verification engine.
// ============================================================================

#include <bits/stdc++.h>
#include <x86intrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

using u64 = uint64_t;
using u8  = uint8_t;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

static u64 SMALL_SIEVE_LIMIT = 1'000'000;
static size_t SIEVE_SCAN_LIMIT = 512;

static constexpr size_t FASTQ_INIT_CAP = 65536;
static constexpr size_t FASTQ_RING_CAP = 12288;
static constexpr size_t FASTQ_WIN_CAP  = 65536;

static constexpr size_t FASTQ_SCAN0 = 32;
static constexpr size_t FASTQ_SCAN1 = 128;
static constexpr size_t FASTQ_SCAN2 = 256;

static constexpr u64 DELTA_WINDOW_NORMAL    = 1'000'000ULL;
static constexpr u64 DELTA_WINDOW_EMERGENCY = 50'000'000ULL;

// ---------------------------------------------------------------------------
// RDTSCP
// ---------------------------------------------------------------------------

static inline u64 rdtscp() {
    unsigned aux;
    return __rdtscp(&aux);
}

// ---------------------------------------------------------------------------
// FastQ duplicate suppression (critical)
// ---------------------------------------------------------------------------

static constexpr size_t SEEN_MASK = 16383; // power of 2
static thread_local array<u64, SEEN_MASK + 1> seen_stamp{};

static inline bool seen(u64 p) {
    size_t h = (p * 11400714819323198485ull) & SEEN_MASK;
    if (seen_stamp[h] == p) return true;
    seen_stamp[h] = p;
    return false;
}

// ---------------------------------------------------------------------------
// Miller–Rabin (64-bit deterministic)
// ---------------------------------------------------------------------------

static inline u64 mul_mod(u64 a, u64 b, u64 m) {
    return (u64)((__uint128_t)a * b % m);
}

static u64 pow_mod(u64 a, u64 d, u64 m) {
    u64 r = 1;
    while (d) {
        if (d & 1) r = mul_mod(r, a, m);
        a = mul_mod(a, a, m);
        d >>= 1;
    }
    return r;
}

static inline bool is_prime64(u64 n) {
    if (n < 2) return false;
    for (u64 p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }

    u64 d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }

    static constexpr u64 bases[] = {
        2ULL, 325ULL, 9375ULL, 28178ULL,
        450775ULL, 9780504ULL, 1795265022ULL
    };

    for (u64 a : bases) {
        if (a % n == 0) continue;
        u64 x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        for (int r = 1; r < s; ++r) {
            x = mul_mod(x, x, n);
            if (x == n - 1) goto cont;
        }
        return false;
        cont:;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Small sieve
// ---------------------------------------------------------------------------

static void build_small_sieve(
    vector<u8>& is_prime_small,
    vector<u64>& primes_24
) {
    const u64 LIM = SMALL_SIEVE_LIMIT;

    is_prime_small.assign(LIM + 1, 1);
    is_prime_small[0] = is_prime_small[1] = 0;

    for (u64 i = 2; i * i <= LIM; ++i)
        if (is_prime_small[i])
            for (u64 j = i * i; j <= LIM; j += i)
                is_prime_small[j] = 0;

    primes_24.reserve(LIM/10);

    u64 cnt = 0;
    for (u64 p = 2; p <= LIM && cnt < SIEVE_SCAN_LIMIT; ++p) {
        if (!is_prime_small[p]) continue;
        if ((p + 2 <= LIM && is_prime_small[p + 2]) ||
            (p + 4 <= LIM && is_prime_small[p + 4])) {
            primes_24.push_back(p);
            
        }
		++cnt;
    }

    cnt = 0;
    for (u64 p = LIM; p > 2 && cnt < SIEVE_SCAN_LIMIT; --p) {
        if (!is_prime_small[p]) continue;
        if ((p > 2 && is_prime_small[p - 2]) ||
            (p > 4 && is_prime_small[p - 4])) {
            primes_24.push_back(p);
           
        }
		 ++cnt;
    }
	
}

// ---------------------------------------------------------------------------
// FastQ (patched)
// ---------------------------------------------------------------------------

struct FastQ {
    vector<u64> pool;
    deque<u64> ring;
    array<u64, FASTQ_WIN_CAP> win{};
    size_t win_pos = 0, win_cnt = 0;

    void init() {
        pool.clear(); pool.reserve(FASTQ_INIT_CAP);
        ring.clear();
        win_pos = win_cnt = 0;
    }

    inline void learn(u64 p) {
        if (!seen(p)) {
            pool.push_back(p);
            ring.push_back(p);
            if (ring.size() > FASTQ_RING_CAP)
                ring.pop_front();
        }
        win[win_pos] = p;
        win_pos = (win_pos + 1) & (FASTQ_WIN_CAP - 1);
        if (win_cnt < FASTQ_WIN_CAP) ++win_cnt;
    }

    template<class F>
    bool scan(size_t k, F&& f) {
        size_t lim = min(k, win_cnt);
        for (size_t i = 0; i < lim; ++i)
            if (f(win[(win_pos + FASTQ_WIN_CAP - 1 - i) & (FASTQ_WIN_CAP - 1)]))
                return true;

        size_t start = pool.size() > k ? pool.size() - k : 0;
        for (size_t i = pool.size(); i-- > start;)
            if (f(pool[i])) return true;

        return false;
    }
};

// ---------------------------------------------------------------------------
// Δd fallback
// ---------------------------------------------------------------------------

static inline bool dd_scan(
    u64 N,
    FastQ& fq,
    u64 win,
    const vector<u8>& is_prime_small,
    u64& last_p,
    bool& have_last
) {
    u64 C = N >> 1;
    u64 d0 = ((C & 1) == 0) ? 1 : 2;

    for (u64 d = d0; d <= win; d += 2) {
        u64 p = C + d;
        u64 q = C - d;
        if (q < 2 || p >= N) break;

        bool p_ok = (p <= SMALL_SIEVE_LIMIT) ? is_prime_small[p] : is_prime64(p);
        if (!p_ok) continue;

        bool q_ok = (q <= SMALL_SIEVE_LIMIT) ? is_prime_small[q] : is_prime64(q);
        if (!q_ok) continue;

        fq.learn(p);
        last_p = p;
        have_last = true;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Emergency fallback
// ---------------------------------------------------------------------------

static inline bool emergency_fallback(
    u64 N,
    FastQ& fq,
    const vector<u8>& is_prime_small,
    const vector<u64>& primes_24,
    u64& last_p,
    bool& have_last
) {
    for (u64 p : primes_24) {
        if (p >= N) break;
        u64 q = N - p;
        bool ok = (q <= SMALL_SIEVE_LIMIT) ? is_prime_small[q] : is_prime64(q);
        if (ok) {
            u64 winner = (p > q) ? p : q;
    fq.learn(winner);
    last_p = winner;
    have_last = true;
    return true;
        }
    }
    return dd_scan(N, fq, DELTA_WINDOW_EMERGENCY, is_prime_small, last_p, have_last);
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

void run(u64 startN, u64 endN, int threads) {

    vector<u8> is_prime_small;
    vector<u64> primes_24;

    u64 range = endN - startN;
    SMALL_SIEVE_LIMIT = max<u64>(300'000, range / 1000);
    SIEVE_SCAN_LIMIT  = SMALL_SIEVE_LIMIT / 2;

    build_small_sieve(is_prime_small, primes_24);

#ifdef _OPENMP
    omp_set_num_threads(max(1, threads));
#endif

    u64 total = 0, ok = 0;
    u64 total_cycles = 0;

    auto t0 = chrono::steady_clock::now();

#pragma omp parallel reduction(+:ok,total,total_cycles)
    {
        FastQ fq;
        fq.init();

        u64 last_p = 0;
        bool have_last = false;

        u64 c0 = rdtscp();

#pragma omp for schedule(static)
        for (u64 N = startN; N <= endN; N += 2) {

            bool found = false;

            auto try_p = [&](u64 p) {
				if ((p & 1) == 0) return false;
                if (p >= N) return false;
                u64 q = N - p;
                bool ok = (q <= SMALL_SIEVE_LIMIT) ? is_prime_small[q] : is_prime64(q);
                if (ok) {
                    u64 winner = (p > q) ? p : q;
    fq.learn(winner);
    last_p = winner;
    have_last = true;
    return true;
                }
                return false;
            };

           if (fq.scan(FASTQ_SCAN2, try_p))
    found = true;

            if (!found)
                for (size_t i = 0; i < primes_24.size() && i < SIEVE_SCAN_LIMIT; ++i)
                    if (try_p(primes_24[i])) { found = true; break; }

            if (!found)
                found = dd_scan(N, fq, DELTA_WINDOW_NORMAL,
                                is_prime_small, last_p, have_last);

            if (!found)
                found = emergency_fallback(N, fq, is_prime_small,
                                           primes_24, last_p, have_last);

            if (found) ++ok;
            ++total;
        }

        u64 c1 = rdtscp();
        total_cycles += (c1 - c0);
    }

    auto t1 = chrono::steady_clock::now();
    double sec = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() / 1000.0;

    printf("[Δ-PROD] Coverage: %llu / %llu\n",
           (unsigned long long)ok,
           (unsigned long long)total);
    printf("[Δ-PROD] Time: %.3f sec\n", sec);
    printf("[Δ-PROD] Cycles/N (total-core): %.1f\n",
           (double)total_cycles / (double)(total ? total : 1));
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s startN endN [threads]\n", argv[0]);
        return 1;
    }

    u64 startN = strtoull(argv[1], nullptr, 10);
    u64 endN   = strtoull(argv[2], nullptr, 10);
    int threads = (argc >= 4) ? atoi(argv[3]) : 1;

    if (startN & 1) ++startN;
    if (endN & 1)   --endN;

    run(startN, endN, threads);
    return 0;
}

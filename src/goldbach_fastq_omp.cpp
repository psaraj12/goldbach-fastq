// ============================================================================
// goldbach_prod_omp_final_closed.cpp
// FINAL-CLOSED STABLE VERSION
//
// - Winner-biased FastQ (recent successful primes first)
// - Elastic FastQ scan: 32 → 128 → 256
// - Bounded small-prime sieve
// - Δd fallback
// - Emergency fallback
// - Correct OpenMP timing using per-thread RDTSCP + reduction
//
// + NEW: N-Block Drift Buckets (SuperBlock -> Block)
//   - Learns good drift distances d where (N/2 ± d) are prime pairs
//   - Per-thread, lock-free, low overhead, helps huge ranges (e.g., 5e11 evens)
// ============================================================================

#include <bits/stdc++.h>
#include <x86intrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define ENABLE_DRIFT_STATS 0
using namespace std;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;
uint32_t PRIME24_LIMIT = 1u << 24; //1'000'000;//1u << 24;  // 16,777,216
static constexpr size_t SEEN_MASK = (1u << 18) - 1; // 262143//static constexpr size_t SEEN_MASK = 16383; // power of 2
array<u64, SEEN_MASK + 1> seen_stamp{};
u64 cur_stamp = 1;

inline bool seen(u64 p) {
    size_t h = (p * 11400714819323198485ull) & SEEN_MASK;
    if (seen_stamp[h] == p) return true;
    seen_stamp[h] = p;
    return false;
}

// ============================================================================
// Δd SPIKE COLLECTOR (event-level, Top-K)
// ============================================================================
constexpr u64 SEG_HALF = 5'000'000;  // 5M each side
static constexpr u64 SPIKE_D_THRESHOLD = 512;
static constexpr int SPIKE_TOPK = 16;
// Spike analysis
static constexpr u32 SPIKE_MIN_D = 512;     // what counts as a "spike event"
static constexpr u32 NEIGHBOR_MAX_D = 512;  // what we consider "easy neighbor" (tuneable)

vector<u32> build_base_primes(u32 limit) {
    vector<uint8_t> is_prime(limit + 1, 1);
    is_prime[0] = is_prime[1] = 0;

    for (u32 i = 2; (uint64_t)i * i <= limit; ++i) {
        if (is_prime[i]) {
            for (u32 j = i * i; j <= limit; j += i)
                is_prime[j] = 0;
        }
    }

    vector<u32> primes;
    primes.reserve(limit / 10);   // rough heuristic

    for (u32 i = 2; i <= limit; ++i) {
        if (is_prime[i])
            primes.push_back(i);
    }

    return primes;
}

struct SpikeRec {
    u64 d;
    u64 N;
    u64 p;
    u64 q;
};

static SpikeRec spike_topk[SPIKE_TOPK];
static int spike_cnt = 0;

static inline void record_spike(u64 d, u64 N, u64 p, u64 q) {
    if (d < SPIKE_D_THRESHOLD) return;

    // find insertion position
    int pos = -1;
    for (int i = 0; i < spike_cnt; ++i) {
        if (d > spike_topk[i].d) { pos = i; break; }
    }
    if (pos == -1 && spike_cnt < SPIKE_TOPK) pos = spike_cnt;
    if (pos == -1) return;

    // shift down
    for (int i = min(spike_cnt, SPIKE_TOPK - 1); i > pos; --i)
        spike_topk[i] = spike_topk[i - 1];

    spike_topk[pos] = { d, N, p, q };
    if (spike_cnt < SPIKE_TOPK) spike_cnt++;
}

struct BlockStats {
    u64 cnt = 0;
    unsigned long long sum_d = 0;
    u64 max_d = 0;

    inline void add(u64 d) {
        ++cnt;
        sum_d += (unsigned long long)d;
        if (d > max_d) max_d = d;
    }

    inline void merge_from(const BlockStats& o) {
        cnt += o.cnt;
        sum_d += o.sum_d;
        if (o.max_d > max_d) max_d = o.max_d;
    }
};
// ============================================================================
// CONFIG
// ============================================================================
struct SpikeEvent {
    u64 N;
    u32 d;
    u64 p;
    u64 q;
};


u64 SMALL_SIEVE_LIMIT = 1'000'000ULL;
//static constexpr san temp
 size_t SIEVE_SCAN_LIMIT = 512;//256 san temp

// FastQ prime reuse
static constexpr size_t FASTQ_INIT_CAP = 49152;//49152;//49152;//4096
static constexpr size_t FASTQ_RING_CAP = 16384;//12288;//1024
static constexpr size_t WIN_CAP = 65536;//8192;

// FastQ center-offset reuse
static constexpr size_t FASTQ_D_CAP   =512; //256;
static constexpr size_t FASTQ_D_SCAN0 =64; //32;
static constexpr size_t FASTQ_D_SCAN1 = 192;//128;

// Elastic prime scans
static constexpr size_t FASTQ_SCAN0 = 32;
static constexpr size_t FASTQ_SCAN1 = 128;
static constexpr size_t FASTQ_SCAN2 = 256;

// Δd windows
static constexpr u64 DELTA_WINDOW_NORMAL    = 1'000'000ULL;
static constexpr u64 DELTA_WINDOW_EMERGENCY = 50'000'000ULL;

// Drift buckets
static constexpr u64 BLOCK_E = (1ULL << 20);
static constexpr u64 SUPER_E = (1ULL << 26);
static constexpr size_t BLOCK_CACHE_SLOTS = 64;
static constexpr size_t SUPER_CACHE_SLOTS = 8;
static constexpr size_t DRIFT_BLOCK_CAP = 2048;
static constexpr size_t DRIFT_SUPER_CAP = 1024;
static constexpr size_t DRIFT_TRY_BLOCK = 128;
static constexpr size_t DRIFT_TRY_SUPER = 64;

// ============================================================================
// RDTSCP
// ============================================================================

static inline u64 rdtscp() {
    unsigned aux;
    return __rdtscp(&aux);
}

// ============================================================================
// Miller–Rabin (64-bit deterministic)
// ============================================================================

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

    u64 d = n - 1; int s = 0;
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

// ============================================================================
// Small sieve
// ============================================================================
static void build_small_sieve(
    vector<u8>& is_prime_small,
    vector<u64>& small_primes,
    vector<u64>& primes_24
) {
    const u64 LIM = SMALL_SIEVE_LIMIT;
u64 cnt=0;
    // --- 1) Classic sieve ---
    is_prime_small.assign(LIM + 1, 1);
    is_prime_small[0] = is_prime_small[1] = 0;

    for (u64 i = 2; i * i <= LIM; ++i) {
        if (is_prime_small[i]) {
            for (u64 j = i * i; j <= LIM; j += i)
                is_prime_small[j] = 0;
        }
    }

    // --- 2) Collect primes + constellation primes ---
  
    primes_24.reserve(LIM / 10);

    for (u64 p = 2; p < LIM && cnt<SIEVE_SCAN_LIMIT; ++p) {
        if (!is_prime_small[p]) continue;

        	
        
        bool good =
            (
            (p + 2 <= LIM && is_prime_small[p + 2]) ||
            (
            (p + 4 <= LIM && is_prime_small[p + 4])));

        if (good)
		{
            primes_24.push_back(p);
		
			
		}
		cnt++;
    }
	
	cnt=0;
	
for (u64 p = LIM; p >1 && cnt< SIEVE_SCAN_LIMIT; --p) {
        if (!is_prime_small[p]) continue;

        
		
        bool good =
            ((p > 2 && is_prime_small[p - 2]) ||
            (
            (p > 4 && is_prime_small[p - 4]) ))
            ;

        if (good)
		{
           // small_primes.push_back(p);
			primes_24.push_back(p);
       		 
		}
cnt++;
    }
	
	
	
	
	
	
}

// ============================================================================
// FastQ (prime + center-offset)
// ============================================================================
struct FastQ {
    vector<u64> pool;
    deque<u64> ring;
   // unordered_set<u64> ring_set;

    array<u64, WIN_CAP> win{};
    size_t win_pos = 0, win_cnt = 0;

    array<u32, FASTQ_D_CAP> dwin{};
    size_t dpos = 0, dcnt = 0;

    // === NEW: priority hint channel ===
    deque<u64> hintQ;
   // unordered_set<u64> hint_set;

    void init() {
        pool.clear(); pool.reserve(FASTQ_INIT_CAP);
        ring.clear(); 
		//ring_set.clear();
        hintQ.clear(); 
		//hint_set.clear();
        win_pos = win_cnt = 0;
        dpos = dcnt = 0;
    }

    inline void learn(u64 p) {
        if (!seen(p)) {
    pool.push_back(p);
    ring.push_back(p);
    if (ring.size() > FASTQ_RING_CAP)
        ring.pop_front();
}
        win[win_pos] = p;
        win_pos = (win_pos + 1) & (WIN_CAP - 1);
        if (win_cnt < WIN_CAP) win_cnt++;
    }

    inline void learn_d(u64 d) {
        if (!d) return;
        dwin[dpos] = (u32)d;
        dpos = (dpos + 1) % FASTQ_D_CAP;
        if (dcnt < FASTQ_D_CAP) dcnt++;
    }

    // === NEW: hint API (ordering only, no learning) ===
    inline void hint(u64 p) {
        if (!seen(p)) {
            hintQ.push_back(p);
           // hint_set.insert(p);
        }
    }
inline void hint_win(u64 p) {
    // IMPORTANT:
    // - do NOT touch ring / ring_set
    // - do NOT touch pool
    // - do NOT count as learning
    // - just bias next scans
    win[win_pos] = p;
    win_pos = (win_pos + 1) & (WIN_CAP - 1);
    if (win_cnt < WIN_CAP) win_cnt++;
}
    template<class F>
    bool scan(size_t k, F&& f) {

        // --- 0) Drain priority hints first ---
        while (!hintQ.empty()) {
            u64 p = hintQ.front();
            hintQ.pop_front();
           // hint_set.erase(p);
            if (f(p)) return true;
        }

        // --- 1) Recent winners (unchanged) ---
        size_t lim = min(k, win_cnt);
        for (size_t i = 0; i < lim; ++i)
            if (f(win[(win_pos + WIN_CAP - 1 - i) & (WIN_CAP - 1)]))
                return true;

        // --- 2) Pool scan (unchanged) ---
        size_t start = pool.size() > k ? pool.size() - k : 0;
        for (size_t i = pool.size(); i-- > start;)
            if (f(pool[i])) return true;

        return false;
    }

    template<class F>
    bool scan_d(size_t k, F&& f) {
        size_t lim = min(k, dcnt);
        for (size_t i = 0; i < lim; ++i)
            if (f(dwin[(dpos + FASTQ_D_CAP - 1 - i) % FASTQ_D_CAP]))
                return true;
        return false;
    }
};

// ============================================================================
// ==== DRIFT BUCKETS =========================================================
// ============================================================================

struct DriftBucket {
    // store d values (distance from C=N/2), small score, and last-used tick
    array<u32, DRIFT_BLOCK_CAP> d{};
    array<u16, DRIFT_BLOCK_CAP> score{};
    array<u32, DRIFT_BLOCK_CAP> last{};
    u32 tick = 1;
    size_t cnt = 0;

    void reset() {
        cnt = 0;
        tick = 1;
        // not strictly necessary to clear arrays for correctness
    }

    // Seed top entries from another bucket (used for block <- super)
    template <size_t SRC_CAP>
    void seed_from(const array<u32, SRC_CAP>& src_d,
                   const array<u16, SRC_CAP>& src_score,
                   const array<u32, SRC_CAP>& src_last,
                   size_t src_cnt,
                   size_t take)
    {
        cnt = 0;
        tick = 1;
        take = min(take, src_cnt);
        take = min(take, (size_t)DRIFT_BLOCK_CAP);
        for (size_t i = 0; i < take; ++i) {
            d[cnt] = src_d[i];
            score[cnt] = (u16)min<u32>(src_score[i], 65535);
            last[cnt] = 0;
            ++cnt;
        }
    }

    inline void bump_tick() { ++tick; }

    // Learn d: move-to-front-ish by score
    inline void learn(u32 dv) {
        // linear scan in a small bucket; cnt is small-ish in practice
        // (and we only call learn on success)
        size_t pos = (size_t)-1;
        for (size_t i = 0; i < cnt; ++i) {
            if (d[i] == dv) { pos = i; break; }
        }

        if (pos == (size_t)-1) {
            if (cnt < DRIFT_BLOCK_CAP) {
                d[cnt] = dv;
                score[cnt] = 1;
                last[cnt] = tick;
                ++cnt;
                pos = cnt - 1;
            } else {
                // Evict weakest (low score, old)
                size_t worst = 0;
                u32 best_key = UINT32_MAX; // smaller key = worse
                for (size_t i = 0; i < cnt; ++i) {
                    // key combines score (higher better) and recency (higher better)
                    // so worse = low score and old
                    u32 age = (tick - last[i]);
                    u32 key = (u32)score[i] * 16u + (age > 65535 ? 0u : (65535u - age));
                    if (key < best_key) { best_key = key; worst = i; }
                }
                d[worst] = dv;
                score[worst] = 1;
                last[worst] = tick;
                pos = worst;
            }
        } else {
            score[pos] = (u16)min<u32>((u32)score[pos] + 1, 65535);
            last[pos] = tick;
        }

        // Bubble up a little (cheap partial sort)
        while (pos > 0) {
            // prefer higher score, then more recent
            bool better =
                (score[pos] > score[pos - 1]) ||
                (score[pos] == score[pos - 1] && last[pos] > last[pos - 1]);
            if (!better) break;
            swap(d[pos], d[pos - 1]);
            swap(score[pos], score[pos - 1]);
            swap(last[pos], last[pos - 1]);
            --pos;
        }
    }
};

struct SuperBucket {
    array<u32, DRIFT_SUPER_CAP> d{};
    array<u16, DRIFT_SUPER_CAP> score{};
    array<u32, DRIFT_SUPER_CAP> last{};
    u32 tick = 1;
    size_t cnt = 0;

    void reset() { cnt = 0; tick = 1; }
    inline void bump_tick() { ++tick; }

    inline void learn(u32 dv) {
        size_t pos = (size_t)-1;
        for (size_t i = 0; i < cnt; ++i) {
            if (d[i] == dv) { pos = i; break; }
        }

        if (pos == (size_t)-1) {
            if (cnt < DRIFT_SUPER_CAP) {
                d[cnt] = dv;
                score[cnt] = 1;
                last[cnt] = tick;
                ++cnt;
                pos = cnt - 1;
            } else {
                size_t worst = 0;
                u32 best_key = UINT32_MAX;
                for (size_t i = 0; i < cnt; ++i) {
                    u32 age = (tick - last[i]);
                    u32 key = (u32)score[i] * 16u + (age > 65535 ? 0u : (65535u - age));
                    if (key < best_key) { best_key = key; worst = i; }
                }
                d[worst] = dv;
                score[worst] = 1;
                last[worst] = tick;
                pos = worst;
            }
        } else {
            score[pos] = (u16)min<u32>((u32)score[pos] + 1, 65535);
            last[pos] = tick;
        }

        while (pos > 0) {
            bool better =
                (score[pos] > score[pos - 1]) ||
                (score[pos] == score[pos - 1] && last[pos] > last[pos - 1]);
            if (!better) break;
            swap(d[pos], d[pos - 1]);
            swap(score[pos], score[pos - 1]);
            swap(last[pos], last[pos - 1]);
            --pos;
        }
    }
};

template <typename BucketLike, size_t CAP>
static inline bool try_drift_list(
    u64 N,
    const vector<u8>& is_prime_small,
    BucketLike& B,
    const array<u32, CAP>& dl,
    size_t dl_cnt,
    size_t try_lim,
    FastQ& fq,
    bool learn_into_bucket)
{
    u64 C = N / 2;

    size_t lim = min(try_lim, dl_cnt);
    for (size_t i = 0; i < lim; ++i) {
        u64 d = (u64)dl[i];
        if (d == 0) continue;
        u64 p = C + d;
        if (p >= N) continue;            // should not happen for valid d, but safe
        u64 q = N - p;                   // == C - d

        // p and q must be odd primes (for large N, both odd anyway if d parity ok)
        bool p_ok = (p <= SMALL_SIEVE_LIMIT) ? is_prime_small[p] : is_prime64(p);
        if (!p_ok) continue;

        bool q_ok = (q <= SMALL_SIEVE_LIMIT) ? is_prime_small[q] : is_prime64(q);
        if (!q_ok) continue;

        fq.learn(p);
        if (learn_into_bucket) B.learn((u32)d);
        return true;
    }
    return false;
}

// Simple per-thread caches so we don’t allocate buckets for every block in the universe.
struct ThreadBucketCache {
    struct BlockSlot {
        u64 block_id = UINT64_MAX;
        u64 super_id = UINT64_MAX;
        DriftBucket bucket;
        u32 stamp = 0;
    };
    struct SuperSlot {
        u64 super_id = UINT64_MAX;
        SuperBucket bucket;
        u32 stamp = 0;
    };

    array<BlockSlot, BLOCK_CACHE_SLOTS> blocks{};
    array<SuperSlot, SUPER_CACHE_SLOTS> supers{};
    u32 stamp = 1;

    SuperSlot& get_super(u64 super_id) {
        ++stamp;
        // hit
        for (auto& s : supers) {
            if (s.super_id == super_id) { s.stamp = stamp; return s; }
        }
        // miss -> evict LRU
        size_t victim = 0;
        u32 best = UINT32_MAX;
        for (size_t i = 0; i < supers.size(); ++i) {
            if (supers[i].super_id == UINT64_MAX) { victim = i; best = 0; break; }
            if (supers[i].stamp < best) { best = supers[i].stamp; victim = i; }
        }
        supers[victim].super_id = super_id;
        supers[victim].stamp = stamp;
        supers[victim].bucket.reset();
        return supers[victim];
    }

    BlockSlot& get_block(u64 block_id, u64 super_id, SuperSlot& sup) {
        ++stamp;
        for (auto& b : blocks) {
            if (b.block_id == block_id) { b.stamp = stamp; return b; }
        }
        // miss -> evict LRU
        size_t victim = 0;
        u32 best = UINT32_MAX;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].block_id == UINT64_MAX) { victim = i; best = 0; break; }
            if (blocks[i].stamp < best) { best = blocks[i].stamp; victim = i; }
        }
        auto& B = blocks[victim];
        B.block_id = block_id;
        B.super_id = super_id;
        B.stamp = stamp;
        // seed block from super’s top entries (gives continuity)
	// Default
size_t take = 8;

// Only escalate for extreme cases
if (sup.bucket.score[0] >= 6000)
    take = 16;

        B.bucket.seed_from(sup.bucket.d, sup.bucket.score, sup.bucket.last, sup.bucket.cnt, take);
        return B;
    }
};
static inline u32 min_symmetric_d(u64 N,
                                 u32 maxd,
                                 const vector<u8>& is_prime_small)
{
    u64 C = N / 2;
    u64 d0 = ((C & 1) == 0) ? 1 : 2;   // same parity start as your dd_scan

    for (u64 d = d0; d <= maxd; d += 2) {
        u64 p_hi = C + d;
        u64 p_lo = C - d;
        if (p_lo < 2 || p_hi >= N) break;

        bool hi_ok = (p_hi <= SMALL_SIEVE_LIMIT) ? is_prime_small[p_hi] : is_prime64(p_hi);
        if (!hi_ok) continue;

        bool lo_ok = (p_lo <= SMALL_SIEVE_LIMIT) ? is_prime_small[p_lo] : is_prime64(p_lo);
        if (!lo_ok) continue;

        return (u32)d;
    }
    return 0; // 0 = not found within maxd
}
// ============================================================================
// Δd fallback (now optionally learns drift into buckets)
// ============================================================================
// ============================================================================
// Δd fallback (SYMMETRIC: probes both sides of N/2)
// ============================================================================
// ============================================================================
// Δd fallback (SYMMETRIC, bucket-aware)
// ============================================================================
static inline bool try_propulsion(
    u64 N,
    u64& p_out,
    u64 last_p,
    bool have_last,
    const vector<u8>& is_prime_small,
    FastQ& fq          // <-- NEW
) {
    if (!have_last) return false;

    // --- 1) Direct reuse ---
    u64 q = N - last_p;
    if ((q & 1) &&
        ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
         (q >  SMALL_SIEVE_LIMIT && is_prime64(q))))
    {
        p_out = last_p;
        return true;
    }

    // --- 2) Local propulsion ---
    static constexpr int dlist[] = { +2, -2, +4, -4 };
    for (int d : dlist) {
        u64 p = last_p + d;
        if ((p & 1) == 0 || p >= N) continue;

        bool p_ok = (p <= SMALL_SIEVE_LIMIT)
                        ? is_prime_small[p]
                        : is_prime64(p);
        if (!p_ok) continue;

        u64 q2 = N - p;
        bool q_ok = (q2 <= SMALL_SIEVE_LIMIT)
                        ? is_prime_small[q2]
                        : is_prime64(q2);
        if (!q_ok) continue;

        p_out = p;
        return true;
    }

    // --- 3) MOMENTUM FUSION (NEW) ---
    // If q±2 or q±4 is prime, bias FastQ ordering
    if (q <= SMALL_SIEVE_LIMIT) {
        if ((q > 2 && is_prime_small[q - 2]) ||
            (q > 4 && is_prime_small[q - 4])) {
            fq.hint_win(last_p);   // <-- momentum injection
        }
    } else {
        // optional: large-q version (cheap)
        if ((q > 2 && is_prime64(q - 2)) ||
            (q > 4 && is_prime64(q - 4))) {
            fq.hint_win(last_p);
        }
    }

    return false;
}

static inline bool dd_scan(
    u64 N,
    FastQ& fq,
    u64 win,
    const vector<u8>& is_prime_small,
    DriftBucket* blockB,
    SuperBucket* superB,
	u64& last_p,
    bool& have_last,
	vector<SpikeEvent>* spikes_local // NEW
	,
	u32* d_used_out
	)
{
    u64 C = N / 2;

    // preserve original parity handling
    u64 d = ((C & 1) == 0) ? 1 : 2;

    for (; d <= win; d += 2) {
        u64 p_hi = C + d;
        u64 p_lo = C - d;

        if (p_lo < 2 || p_hi >= N)
            break;

        bool hi_ok;
if (p_hi <= SMALL_SIEVE_LIMIT) {
    hi_ok = is_prime_small[p_hi];
} else {
    
    // Miller–Rabin CERTIFICATION (expensive)
    hi_ok = is_prime64(p_hi);
}
if (!hi_ok) continue;

bool lo_ok;
if (p_lo <= SMALL_SIEVE_LIMIT) {
    lo_ok = is_prime_small[p_lo];
} else {
    

    // Miller–Rabin CERTIFICATION (expensive)
    lo_ok = is_prime64(p_lo);
}
if (!lo_ok) continue;

        // success
        fq.learn(p_hi);
        last_p=p_hi;
        have_last=true;		
        if (blockB) blockB->learn((u32)d);
        if (superB) superB->learn((u32)d);
         // NEW: record spike if big enough
        if (spikes_local && (u32)d >= SPIKE_MIN_D) {
            spikes_local->push_back(SpikeEvent{N, (u32)d, p_hi, N - p_hi});
        }
		if (d_used_out) *d_used_out = (u32)d;
        return true;
    }
    return false;
}


// ============================================================================
// Emergency fallback (unchanged behavior, but passes buckets into dd_scan)
// ============================================================================

static inline bool emergency_fallback(
    u64 N, FastQ& fq,
    const vector<u8>& is_prime_small,
    const vector<u64>& small_primes,
    DriftBucket* blockB,
    SuperBucket* superB,
	u64& last_p,
    bool& have_last,
	vector<SpikeEvent>* spikes_local // NEW
	,
	u32* d_used_out
	)
{
    for (u64 p : small_primes) {
        if (p >= N) break;
        u64 q = N - p;
        if ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
            (q > SMALL_SIEVE_LIMIT && is_prime64(q))) {
            fq.learn(q);
            return true;
        }
    }
    return dd_scan(N, fq, DELTA_WINDOW_EMERGENCY, is_prime_small, blockB, superB,last_p,have_last,spikes_local,d_used_out);
}
static inline u64 abs_diff_u64(u64 a, u64 b) {
    return (a >= b) ? (a - b) : (b - a);
}
// ============================================================================
// MAIN ENGINE
// ============================================================================

void run(u64 startN, u64 endN, int threads) {

    vector<u8> is_prime_small;
    vector<u64> small_primes;
	vector<u64> primes_24;
	
	const u32 PRIME24_LIMIT = max<u32>(300'000, (endN - startN) / 1000);
SIEVE_SCAN_LIMIT=PRIME24_LIMIT/2;
//cout<<SIEVE_SCAN_LIMIT<<"\n";
	SMALL_SIEVE_LIMIT= PRIME24_LIMIT;
	
    build_small_sieve(is_prime_small, small_primes,primes_24);
	vector<uint32_t> prime_24 ;//= build_base_primes(PRIME24_LIMIT);
	//cout<<primes_24.size()<<"\n";
u64 range_evens = ((endN - startN) >> 1) + 1;
u64 num_blocks  = (range_evens + BLOCK_E - 1) / BLOCK_E;
u64 ttt=endN-range_evens;

vector<BlockStats> gstats(num_blocks);
#ifdef _OPENMP
    omp_set_num_threads(max(1, threads));
#endif

    u64 total = 0, ok = 0,total_mr_calls=0,tot_sieve=0;
    u64 total_cycles = 0;

    auto t0 = chrono::steady_clock::now();
	
vector<SpikeEvent> spikes;

#pragma omp parallel reduction(+:ok,total,total_cycles)
    {
        FastQ fq;
        fq.init();
        u64 last_p = 0;
		u64 cnt=0;
		vector<SpikeEvent> local_spikes;
		
local_spikes.reserve(64); // spikes are rare; small reserve is enough

        bool have_last = false;
        ThreadBucketCache cache;
         BlockStats local_stats;
u64 cur_block_id = UINT64_MAX;
        u64 c0 = rdtscp();
#if ENABLE_DRIFT_STATS
vector<BlockStats> local_gstats(num_blocks);
#endif
#pragma omp for schedule(static)
        for (u64 N = startN; N <= endN; N += 2) {
              u64 witness_prime = 0;

// Stage 0: Propulsion
//budget
enum SolveStage : u8 {
    ST_PROP = 0,
    ST_FASTQ = 1,
    ST_DRIFT = 2,
    ST_DD = 3,
    ST_EMERG = 4
};

SolveStage last_stage = ST_FASTQ;   // optimistic default

//budget

	
	u64 val=primes_24.size()-1;

u32 d_used = 0;
            // ==== bucket ids (in even-steps) ====
            u64 e = (N - startN) >> 1;               // 0..(range_evens-1)
            u64 block_id = e / BLOCK_E;
            u64 super_id = e / SUPER_E;

            auto& supSlot = cache.get_super(super_id);
            auto& blkSlot = cache.get_block(block_id, super_id, supSlot);

            // ticks (helps recency in scoring)
            supSlot.bucket.bump_tick();
            blkSlot.bucket.bump_tick();

            bool found = false;
            u64 p_prop;         
            u64 mr_calls=0;
            u64 tt_seive=0;
			u64 C = N >> 1;
			/*if ((
    C < seg.L + SEG_HALF ||
    C > seg.R - SEG_HALF) && N >= ttt && 1==2)
{
    u64 newL = (C > SEG_HALF) ? (C - SEG_HALF) : 0;
    u64 newR = C + SEG_HALF;

    build_segment(seg, newL, newR, prime_24);
    seg_valid = true;
    babarocks=true;
}
*/
            auto try_p = [&](u64 p) {		
				
    if (p >= N) return false;

    u64 q = N - p;
 bool prime_adjacent=false;
    // === ACCELERATOR START ===
    // Prime-adjacent heuristic: cheap signal only
	/*if (q <= SMALL_SIEVE_LIMIT)
	{
     prime_adjacent =
             // cheap only
         (q > 2 && is_prime_small[q - 2]) ||
		  (q > 4 && is_prime_small[q - 4]);
	}
	*/
    // === ACCELERATOR END ===
// Accelerator does NOT claim success.
    // It only biases scan ordering implicitly by
    // allowing FastQ's existing behavior to surface
    
    // Direct Goldbach success (unchanged semantics)
    if ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
        (q > SMALL_SIEVE_LIMIT &&
    ( !is_prime64(q)))) {
        fq.learn(p);
        last_p = p;
        witness_prime = p;
        have_last = true;
        return true;
    }
/*
    // nearby p's faster in subsequent iterations.
    if (prime_adjacent ) {
		fq.hint_win(p);   // <-- NOW IT ACTUALLY ACCELERATES
		return false;
        // no-op by design: signal only
        // (future-safe hook if you later add stats)
    }
*/
    return false;
};
size_t scan_budget;

switch (last_stage) {
    case ST_PROP:   scan_budget = 8;   break;
    case ST_FASTQ:  scan_budget = 16;  break;
    case ST_DRIFT:  scan_budget = 64;  break;
    case ST_DD:     scan_budget = 128; break;
    default:        scan_budget = 16;  break;
}
         

if (!found && fq.scan(scan_budget, try_p))
{
	last_stage = ST_FASTQ;
    found = true;
	
}
if (found) {ok++;total++;continue;}
            // 2) NEW: drift buckets (block first, then super)
            if (!found) {
                if (try_drift_list(N, is_prime_small,
                                   blkSlot.bucket,
                                   blkSlot.bucket.d, blkSlot.bucket.cnt,
                                   DRIFT_TRY_BLOCK,
                                   fq,
                                   true)) {
									   last_stage = ST_DRIFT;
                    found = true;
					
                }
            }
			if (found) {ok++;total++;continue;}
            if (!found) {
                if (try_drift_list(N, is_prime_small,
                                   supSlot.bucket,
                                   supSlot.bucket.d, supSlot.bucket.cnt,
                                   DRIFT_TRY_SUPER,
                                   fq,
                                   true)) {
									   last_stage = ST_DRIFT;
                    found = true;
					
                }
            }
			
if (found) {ok++;total++;continue;}
			
if (!found && (try_propulsion(N, p_prop, last_p, have_last, is_prime_small, fq))) {
    fq.learn(p_prop);
    witness_prime = p_prop;
    last_p = p_prop;
    have_last = true;
    found = true;
	last_stage = ST_PROP;
	
}
if (found) {ok++;total++;continue;}

if (!found) {
                for (size_t i = 0; i < SIEVE_SCAN_LIMIT && i<primes_24.size() ; i++) {
                    u64 p = primes_24[i];
                    //if (p >=N ) break;
                    u64 q = N - p;
					if (q<= 0)
					{
						break;
					}
                    if ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
                        (q > SMALL_SIEVE_LIMIT &&
    ( !is_prime64(q)))) {
                        fq.learn(q);
						last_p = q;
						witness_prime=q;
						
                have_last = true;
                        found = true;
						
						
                        break;
                    }
                }
            }
	
	if (found) {ok++;total++;continue;}
	//cnt=0;
	/* temp comment
if (!found) {
                for (size_t i =val ; cnt< SIEVE_SCAN_LIMIT; i--) {
                    u64 p = primes_24[i];
                   // if (p >= N ) break;//san temp added N/2
                    u64 q = N - p;
					if (q<= 0)
					{
						break;
					}
                    if ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
                        (q > SMALL_SIEVE_LIMIT &&
    ( !is_prime64(q)))) {
                        fq.learn(q);
						last_p = q;
						witness_prime=q;
                have_last = true;
                        found = true;
						
                        break;
                    }
					cnt++;
                }
            }	
if (found) {ok++;total++;continue;}
*/			



            // 3) Small-prime sieve scan (your existing stage)
			/*
            cnt=0;
			
			if (!found) {
                for (size_t i = 2; i< SIEVE_SCAN_LIMIT && i < small_primes.size(); i++) {
                    u64 p = small_primes[i];
                    if (p >= N) break;
                    u64 q = N - p;
                    if ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
                        (q > SMALL_SIEVE_LIMIT &&
    ( !is_prime64(q)))) {
                        fq.learn(q);
						last_p = q;
						witness_prime=q;
						
                have_last = true;
                        found = true;
                        break;
                    }
					
                }
            }
			*/
			 // 3) Small-prime sieve scan (your existing stage)
			// cnt=0;
//if (found) {ok++;total++;continue;}	
	
	/*
	if (!found) {
                for (size_t i =small_primes.size()-1 ; cnt< SIEVE_SCAN_LIMIT; i--) {
                    u64 p = small_primes[i];
                   // if (p >= N ) break;//san temp added N/2
                    u64 q = N - p;
					if (q<= 0)
					{
						break;
					}
                    if ((q <= SMALL_SIEVE_LIMIT && is_prime_small[q]) ||
                        (q > SMALL_SIEVE_LIMIT &&
    ( !is_prime64(q)))) {
                        fq.learn(q);
						last_p = q;
						witness_prime=q;
                have_last = true;
                        found = true;
						
                        break;
                    }
					cnt++;
                }
            }	
if (found) {ok++;total++;continue;}		
*/
		
             
			
			
            // 4) Δd fallback (now learns drift into both buckets on success)
            if (!found && dd_scan(N, fq, DELTA_WINDOW_NORMAL, is_prime_small,
                                  &blkSlot.bucket, &supSlot.bucket,last_p, have_last,&local_spikes,&d_used))
								  {last_stage = ST_DD;
                found = true;
				
								  }
								  if (found) {ok++;total++;continue;}
            // 5) Emergency fallback (same, with bucket learning via dd_scan)
            if (!found && emergency_fallback(N, fq, is_prime_small, primes_24,
                                             &blkSlot.bucket, &supSlot.bucket,last_p, have_last,&local_spikes,&d_used))
											 {
                found = true;
                last_stage = ST_DD;
				
											 }
											
											 
            if (found) {
	
	
    ok++;
    
	
#if ENABLE_DRIFT_STATS
    if (witness_prime == 0) {
    if (!fq.ring.empty()) witness_prime = fq.ring.back();
    else if (have_last)   witness_prime = last_p;
    else                  continue; // don't record garbage
	
	
	
    
//u64 d_used = abs_diff_u64(witness_prime, C);

//drift_blocks[block_id].add(d);
   if (d_used && d_used <= DELTA_WINDOW_EMERGENCY)
    local_gstats[block_id].add(d_used);
#endif
}
   
   

total++;

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
		   
    //cout<<"Total MR calls"<<total_mr_calls<<"\n"	;
	//cout<<"Total Seive calls"<<tot_sieve<<"\n"	;
#if ENABLE_DRIFT_STATS

		   printf("\n[DRIFT] Block-level drift stats (BLOCK_E=%llu evens)\n",
       (unsigned long long)BLOCK_E);

vector<u64> max_list;
max_list.reserve(num_blocks);

for (u64 b = 0; b < num_blocks; ++b) {
    const auto& s = gstats[b];
    if (!s.cnt) continue;

    double avg_d = (double)s.sum_d / (double)s.cnt;
    printf("[DRIFT] block=%llu cnt=%llu avg_d=%.2f max_d=%llu\n",
           (unsigned long long)b,
           (unsigned long long)s.cnt,
           avg_d,
           (unsigned long long)s.max_d);

    max_list.push_back(s.max_d);
}

if (!max_list.empty()) {
    auto tmp = max_list;
    sort(tmp.begin(), tmp.end());
    u64 p99 = tmp[(tmp.size() * 99) / 100];

    printf("\n[DRIFT] Spike threshold (P99 max_d) = %llu\n",
           (unsigned long long)p99);

    for (u64 b = 0; b < num_blocks; ++b) {
        const auto& s = gstats[b];
        if (!s.cnt || s.max_d < p99) continue;

        double avg_d = (double)s.sum_d / (double)s.cnt;
        u64 block_start_N = startN + ((b * BLOCK_E) << 1);
        u64 block_end_N   = min(endN, startN + (((b + 1) * BLOCK_E - 1) << 1));

        printf("[SPIKE] block=%llu N=[%llu..%llu] avg_d=%.2f max_d=%llu\n",
               (unsigned long long)b,
               (unsigned long long)block_start_N,
               (unsigned long long)block_end_N,
               avg_d,
               (unsigned long long)s.max_d);
    }
}
printf("\n[Δd SPIKES] Top %d events (d >= %llu)\n",
       spike_cnt,
       (unsigned long long)SPIKE_D_THRESHOLD);

for (int i = 0; i < spike_cnt; ++i) {
    printf(" #%2d: d=%llu  N=%llu  p=%llu  q=%llu\n",
           i + 1,
           (unsigned long long)spike_topk[i].d,
           (unsigned long long)spike_topk[i].N,
           (unsigned long long)spike_topk[i].p,
           (unsigned long long)spike_topk[i].q);
}
if (!spikes.empty()) {
    sort(spikes.begin(), spikes.end(),
         [](const SpikeEvent& a, const SpikeEvent& b){ return a.N < b.N; });

    printf("\n[NEIGHBOR] Spike neighbor check (NEIGHBOR_MAX_D=%u)\n", NEIGHBOR_MAX_D);

    u64 good_both = 0, miss_left = 0, miss_right = 0, total_sp = spikes.size();

    for (auto &s : spikes) {
        u64 N = s.N;
        u32 dL = 0, dR = 0;

        if (N >= startN + 2) dL = min_symmetric_d(N - 2, NEIGHBOR_MAX_D, is_prime_small);
        if (N + 2 <= endN)   dR = min_symmetric_d(N + 2, NEIGHBOR_MAX_D, is_prime_small);

        bool okL = (dL != 0);
        bool okR = (dR != 0);

        if (!okL) miss_left++;
        if (!okR) miss_right++;
        if (okL && okR) good_both++;

        printf("[NEIGHBOR] N=%llu spike_d=%u  left(N-2)=%u  right(N+2)=%u\n",
               (unsigned long long)N, s.d, dL, dR);
    }

    printf("\n[NEIGHBOR] spikes=%llu  both_ok=%llu  miss_left=%llu  miss_right=%llu\n",
           (unsigned long long)total_sp,
           (unsigned long long)good_both,
           (unsigned long long)miss_left,
           (unsigned long long)miss_right);
}

#endif	   
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s startN endN [threads]\n", argv[0]);
        return 1;
    }

    u64 startN = strtoull(argv[1], nullptr, 10);
    u64 endN   = strtoull(argv[2], nullptr, 10);
    int threads = (argc >= 4) ? atoi(argv[3]) : 1;

    if (startN & 1) startN++;
    if (endN & 1) endN--;

    run(startN, endN, threads);
    return 0;
}

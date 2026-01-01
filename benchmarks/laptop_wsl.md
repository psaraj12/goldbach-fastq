# Benchmark: ASUS Vivobook (WSL2)

CPU: Intel 12th-gen mobile  
OS: Windows 11 + WSL2 (Ubuntu)  
Compiler: g++ -O3 -march=native -fopenmp

Range:
[10^18 .. 10^18 + 10^9]

Threads: 24
./goldbachv2 1000000000000000000 1000000001000000000 24
[Δ-PROD] Coverage: 500000001 / 500000001
[Δ-PROD] Time: 38.720 sec
[Δ-PROD] Cycles/N (total-core): 4570.7
Note:
- RDTSCP timing under WSL may be noisy.
- Wall time is the more reliable metric.

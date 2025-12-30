# Benchmark: ASUS Vivobook (WSL2)

CPU: Intel 12th-gen mobile  
OS: Windows 11 + WSL2 (Ubuntu)  
Compiler: g++ -O3 -march=native -fopenmp

Range:
[10^18 .. 10^18 + 10^6]

Threads: 24

Coverage: 500,001 / 500,001  
Time: ~0.016 sec  
Cycles/N: ~650 (total-core, RDTSCP-based)

Note:
- RDTSCP timing under WSL may be noisy.
- Wall time is the more reliable metric.

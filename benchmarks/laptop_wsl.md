# Benchmark: ASUS Vivobook (WSL2)

CPU: Intel 12th-gen mobile  
OS: Windows 11 + WSL2 (Ubuntu)  
Compiler: g++ -O3 -march=native -fopenmp

Range:
[10^18 .. 10^18 + 10^10]

Threads: 24

[13:31:07] santhiagu@LAPTOP-1PN61UJD:/mnt/f$ ./goldbachv1 1000000000000000000 1000000010000000000 24
[Δ-PROD] Coverage: 5000000001 / 5000000001
[Δ-PROD] Time: 13.713 sec
[Δ-PROD] Cycles/N (total-core): 169.2

Note:
- RDTSCP timing under WSL may be noisy.
- Wall time is the more reliable metric.

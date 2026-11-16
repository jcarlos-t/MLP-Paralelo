
# An analytic performance model for overlapping

# execution of memory-bound loop kernels on

# multicore CPUs

#### Ayesha Afzal, Georg Hager, Gerhard Wellein

#### Department of High Performance Computing

#### Erlangen Regional Computing Center

#### Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany

#### {ayesha.afzal,georg.hager,gerhard.wellein}@fau.de

```
Abstract—Complex applications running on multicore proces-
sors show a rich performance phenomenology. The growing
number of cores per ccNUMA domain complicates performance
analysis of memory-bound code since system noise, load imbal-
ance, or task-based programming models can lead to thread
desynchronization. Hence, the simplifying assumption that all
cores execute the same loop can not be upheld. Motivated by
observations on plain and modified versions of the HPCG bench-
mark, we construct a performance model of execution of memory-
bound loop kernels. It can predict the memory bandwidth share
per kernel on a memory contention domain depending on the
number of active cores and which other workload the kernel
is paired with. The only code features required are the single-
thread cache line access frequency per kernel, which is directly
related to the single-thread memory bandwidth, and its saturated
bandwidth. It can either be measured directly or predicted using
the Execution-Cache-Memory (ECM) performance model. The
computational intensity of the kernels and the detailed structure
of the code is of no significance. We validate our model on
Intel Broadwell, Intel Cascade Lake, and AMD Rome processors
pairing various streaming and stencil kernels. The error in
predicting the bandwidth share per kernel is less than 8%.
```
##### I. INTRODUCTION

```
With the number of cores and the peak performance of
modern multicore chips still growing, the memory bandwidth
bottleneck is becoming more severe. Many algorithms in
computational science are based on building blocks that show
memory-bound behavior, i.e, whose performance does not scale
but saturate with respect to the number of active cores when
running on a memory contention domain (usually a ccNUMA
domain). Performance saturation due to bandwidth saturation
should be regarded as a sign that a code is fast enough to
address an architectural bottleneck, so it is not a bad thing
in general. It also opens a clear optimization path via the
reduction of data transfers.
Beyond the simple saturation pattern, however, memory-
bound code exhibits other, more interesting phenomenology. As
was shown in [2], bulk-synchronous barrier-free MPI programs
can showdesynchronizationon a contention domain, i.e.,
processes move away from the initial lockstep state into a state
where execution of loops kernels overlaps with communication
or idleness, leading to automatic communication hiding. This
can be provoked by a deliberate injection of delays, but it can
```
```
also occur automatically by natural system noise and small
load imbalances.
Although it is often observed in programs that have a
significant communication overhead, desynchronization is not
limited to this scenario. In fact, barrier-free code (and also
modern, task-based programming models) allows for concurrent
execution of code with very different characteristics on a
contention domain. In this paper we investigate a specific
scenario: concurrent execution of two different loop kernels
onncores each. We construct a performance model that can
describe accurately the memory bandwidth share that each
kernel attains. The model is thus able to predict whether
desynchronization of back-to-back parallel kernels and the
ensuing overlap will speed up or slow down the execution of
threads compared to a purely homogeneous execution. It can
also predict if the mutual overlap will amplify or reduce the
desynchronization effect.
This paper is organized as follows. The rest of this section
covers related work and a motivational example (HPCG). In
Sect. II we describe the experimental setup and methodology,
and Sect. III defines code and performance metrics. The
performance model is developed in IV and validated in Sect. V,
where we also discuss the connection to desynchronization
and close the loop to the initial HPCG example. We give a
summary and an outlook to future work in Sect. VI.
```
```
A. Motivation and related work
This investigation is based on prior work by Afzal et al. [1,
3, 2], who studied idle wave propagation and computational
wave formation in parallel programs on a phenomenological
level. An analytic model of overlapping computational kernels
and communication time on a contention domain was lacking,
however. Alappat et al. [4] observed the consequences of
desynchronization in the context of the well-known HPCG
benchmark:^1 While validating their Roofline model for the
MPI-parallel HPCG, they observed that the DDOT2 (dot
products+=a[i]*b[i]) kernels were in fact faster than
what the local memory bandwidth would allow for. The
authors attributed this to process desynchronization during the
preceding sparse matrix-vector multiplication (SpMV) kernel
```
(^1) https://www.hpcg-benchmark.org/

## arXiv:2011.00243v1 [cs.DC] 31 Oct 2020


```
0
3
6
9
```
```
Rank
```
```
(a) BDW-2 - CoD on
0
5
10
15
19
```
```
(b) CLX - SNC off
```
```
0. 29 0. 295 0. 3 0. 305
6
```
```
1
```
```
Time [s]
```
##### R

```
PERMUTE
```
```
symGS
DDOT2 Allreduce
0. 595 0. 6 0. 605 0. 61
16
```
```
15
```
```
Time [s]
```
```
symGS
DDOT
Allreduce
```
```
161 2 7136173119 8101918012145 415
```
```
7
```
```
9
```
```
11
```
```
13
```
```
Permuted rank,RPERMUTE
```
```
Run time
```
```
[ms
```
##### ]

##### CLX

```
0
```
```
5
```
```
10
```
```
15
```
```
Rank
```
```
6 2 8 4 3 7 0 5 1
```
##### BDW-

```
0
```
```
2
```
```
4
```
```
6
```
```
8
```
```
(c) Duration of DDOT2 (proceeding symGS) in one iteration
```
```
Fig. 1: MPI-only HPCG traces on one ccNUMA domain of Intel
Broadwell (BDW-2) and Cascade Lake (CLX) at a problem size
of 1603 per process. Colors denote SymGS (orange), DDOT
(pink), andMPI_Allreduce(green). Top panels in (a) and
(b): timeline snippets from around a time stamp of 1628 s (x
axes show offsets from this point). Bottom panels in (a) and
(b): same data but with ranks sorted by the starting time of the
DDOT2 kernel. (c) DDOT2 runtime per rank, same sorting as
in bottom panels of (a) and (b).
```
```
and assumed that MPI processes that started the DDOT2 kernel
early could benefit from immediate cache reuse, which was
backed by a measured computational intensity that was higher
than expected. However, it was unclear whether cache reuse was
the only reason for the observed elevated DDOT2 performance.
The HPCG algorithm comprises the following kernels: six of
BLAS-1 type (two DDOT2, one DDOT1 [s+=a[i]*a[i]]
and three DAXPY [a[i]+=s*b[i]]) and one SpMV. Addi-
tionally, the multigrid preconditioner compromises five kernels:
restriction, prolongation, SpMV, and two symmetric Gauss-
Seidel routines (SymGS) as pre- and post-smoothers, each with
a forward and a backward sweep, for coarsening and refinement.
```
```
HPCG has two types of MPI communication: a global collective
operation (MPI_Allreduce) after each vector dot product
and nonblocking point-to-point communication in SpMV and
SymGS.
Figure 1(a) (top) shows a snippet from a timeline diagram
of the HPCG benchmark (problem size 1603 per process) on
the nine cores of one ccNUMA domain of an Intel Broadwell
processor (see Sect. II for hardware details and software setup).
The colors encode the SymGS kernel (orange), the DDOT
kernel (pink), and theMPI_Allreducecall (green). Note
that this is only a part of a single iteration; all effects described
here occur in every iteration of the algorithm.
The runtime of the SymGS kernel is about 20 times
longer than that of DDOT2 here, so the clearly visible
desynchronization effect has a minor influence on the SymGS
performance. However, the execution of DDOT2 is strongly
out of sync across processes. “Early” ranks still overlap with
SymGS running on other cores, while “late” ranks overlap
with waiting time (i.e., idleness) inMPI_Allreduce. The
former must compete for memory bandwidth with SymGS and
thus take longer to execute, while the latter can utilize more
bandwidth per core, making them execute faster. This can be
seen better in the lower part of Fig. 1(a), where we have sorted
the MPI ranks according to the starting time of the DDOT
kernel from early (bottom) to late (top). The late starters clearly
take less time than the early starters, although the latter are
those with cache reuse potential. In Fig. 1(b), the experiment
was repeated on a 20-core Intel Cascade Lake processor. The
effect is less pronounced here since the DDOT2 runtime is
now significantly longer than the desynchronization time scale.
There is hence less opportunity for DDOT2 to overlap with
MPI_Allreduce.
Figure 1(c) shows the runtime of the DDOT2 kernel per
rank in both experiments, using the same sorting as in (a)
and (b). The runtimes are monotonically decreasing in both
cases, showing clearly that late starters have better performance.
In summary, the desynchronization of the SymGS kernel, in
itself a negligible effect, leads to some cores executing the
DDOT2 kernel faster due to overlapping with idleness in the
MPI_Allreducefunction. This is the real reason for the
“faster-than-light” performance observed in [4].
In the broader context of desynchronized execution of barrier-
free bulk-synchronous code, the general question arises how
```
```
Exec_A
```
```
Exec_B
```
```
Exec_B
```
```
Exec_A
```
```
Exec_A
```
```
Comm_B
```
```
Comm_B
```
```
Exec_A
```
```
Exec_A
```
```
Idle_B
```
```
Idle_B
```
```
Exec_A
```
a1. Identical kernels: Exec_B1 := Exec_A
a2. Different kernels: Exec_A 6 =Exec_B

```
b1. Latency-limited communication
b2. Bandwidth-limited communication
```
```
P
```
```
P
```
```
Time Time Time
```
```
(c)
```
```
Fig. 2: Characterization of concurrency scenarios for bandwidth-limited computational kernels in parallel programs. “P0” and
“P1” denote multi-threaded processes or groups of MPI processes within a contention domain. Each thread or process in a group
executes the same code on different data. (a1, a2) Overlapping memory-bound loop kernels of different (or same) characteristics,
(b1, b2) overlapping loop kernel execution with latency- or bandwidth-bound communication, and (c) overlapping loop kernel
execution with idleness or code that addresses nonshared or scalable resources.
```

```
0
5
10
15
19
```
```
Rank
```
```
(a) DDOT
0
5
10
15
19
```
```
(b) DDOT2 and DDOT
```
(^10). 59 0. 6 0. 61
20
Time [s]

##### N

```
AMASS
```
```
symGS
DDOT
SpMV
```
(^10). 9 0. 92 0. 94 0. 96
20
Time [s]
SpMV
DDOT
DAXPY+DAXPY
DDOT
6
8
10
12
14
Permuted rank,RPERMUTE
Run time [ms]
DDOT
0
5
10
15
Rank
(^216176131781191910318121415405) DDOT
(^919181811216761013171215035144)
(^918191216118761013171215053144) DDOT
(c) DDOTx (proceeding symGS, SpMV and DAXPY) in one iteration
Fig. 3: Top panels of (a) and (b): Timeline snippets of a
modified HPCG variant (missing any reduction operations) on
an Intel CLX socket at a problem size of 1603 per process,
showing (a) DDOT2 between SymGS and SpMV and (b)
DDOT2 and DDOT1 with two DAXPYs in between. The
bottom panels of (a) and (b) show quantitative timelines of
the number of ranks concurrently executing the DDOT kernels.
Dark red color in (a) denotes time spent inMPI_Waitfor
nonblocking point-to-point communication during SpMVM. (c)
shows time spent in the DDOTx kernels per core vs. permuted
rank (permutation as in Fig. 1).
much bandwidth and thus performance can be achieved per core
if different computational kernels are executed concurrently by
cores on the same contention domain, if part of the cores are
idle, or if they spend time communicating. Figure 2 shows the
range of possible scenarios.
While scenario (b), i.e., overlapping of loop kernel execution
with communication, was covered in [2], and scenario (c) is
well described by the Execution-Cache-Memory performance
model for multicore processors [7], scenario (a) was not studied
before.
To further set the stage and motivate the relevance of
the bandwidth sharing phenomenon, we conducted further
experiments with a modified variant of HPCG that is identical
to the original except for the lack ofMPI_Allreducecalls.
This allows for desynchronized states to survive for a long
time, since no other global MPI operations are used in HPCG.
We also executed a single iteration of the benchmark only.
Figures 3 (a) and (b) shows timeline snippets of the modified
HPCG on the same Intel Cascade Lake CPU as before. In
Fig. 3(a) the DDOT2 kernel is sandwiched between a SymGS
and an SpMV, while in Fig. 3(b) we show a broader view of
the two other DDOT kernels, with two DAXPY operations in
between. As all global reductions were removed, the DDOTs
are allowed to overlap with subsequent kernels.
In the bottom panels of (a) and (b) we show quantitative
timelines that tell how many ranks execute the DDOT kernel at
each point in time. This makes it easy to spot phases of homo-
geneous (i.e, single-kernel) execution and desynchronization.
Qualitatively different behavior across the different DDOT
kernels can be observed: In (a), where the tail end of the
DDOT2 execution overlaps with a subsequent SpMV and
its significantMPI_Waittime, the spread in endpoints of
DDOT2 execution is smaller than the spread in their starting
points. This is plausible because, as described earlier, early
starters of DDOT2 take longer while late starters are faster.
The effect can be quantified by a negative skewness parameter
of the accumulated DDOT2 time distribution of− 0 .27 ms. The
second DDOT2, however, is sandwiched between SpMV and
DAXPY, so any overlap with idleness is ruled out. Instead, the
interaction with the low-intensity DAXPY code seems to boost
the desynchronization effect, i.e., the tail end of the accumulated
DDOT2 time distribution is longer than the front end (skewness
0 .42 ms). The last DDOT, which is actually a vector norm
computation (DDOT1), shows a similar characteristic with an
even larger positive skewness of 1.0 ms.
The skewness parameterizes the asymmetry in the distri-
bution of a random variable, but the timeline data from the
HPCG is not normally distributed even if the skewness is
zero. However, we can use the skewness here to identify an
important property of barrier-free parallel loop kernel execution:
Negative skewness indicatesresynchronization, while positive
skewness indicatesdesynchronization. Which of the two occurs
is obviously a function of which other workloads the kernel
is embedded in. Note that the described effects in HPCG,
even in our synchronization-free variant, are small in absolute
numbers since the HPCG runtime is dominated by the multigrid
and SpMV kernels. However, our observations open interesting
questions towards the onset and mitigation of desynchronization
effects in barrier-free memory-bound MPI programs. The goal
of this paper is to establish a performance model for loop
kernels of different characteristics running concurrently on a
contention domain that can describe this scenario with sufficient
accuracy. As an aditional benefit, such a model can also predict
the dynamics of task-based programming models on memory-
bound code.
B. Contributions
This paper makes the following new contributions. (i)
We identify the influence factors that govern the bandwidth
share for individual threads in a “hybrid execution” setting,
specifically two (groups of) threads running loops with different
characteristics on a memory bandwidth contention domain: the
“memory request fraction” to the memory interface and the
saturated memory bandwidth of the different kernels. (ii) We
employ ideas from the ECM performance model to predict the
bandwidth share of different kernels on a contention domain.
(iii) We validate the model for a set of 30 kernel pairings on four
x86 architectures. Despite sigificant differences in hardware


```
TABLE I: Key hardware and software specifications of systems.
Systems BDW-1 BDW-2 CLX Rome
Processor Broadwell EP Broadwell EP Cascade Lake SP Zen
Model Intel Xeon E5-2630 v4 Intel Xeon E5-2697 v4 Intel Xeon Gold 6248 ARM Epyc 7451
Base clock speed 2 .2 GHz 2 .3 GHz 2 .5 GHz 2 .35 GHz
Cores per NUMA domain (SMT) 10 (20) 18 (36) 20 (40) 8 (16)
Private L1 size 32 (×10) KiB 32 (×18) KiB 32 (×20) KiB 32 (×8) KiB
Private L2 size 256 (×10) KiB 256 (×18) KiB 1048 (×20) KiB 512 (×8) KiB
Shared LLC size 25 MiB (10× 2 .5 MiB) 45 MiB (18× 2 .5 MiB) 27 .5 MiB (20× 1 .375 MiB) 8 MiB (per 4 cores)
Memory per node (type) 64 GiB (DDR4) 384 GiB (DDR4) 128 GiB (DDR4) 128 GiB (DDR4)
LD/ST throughput (SIMD) 2/1 (AVX2/FMA3) 2/1 (AVX2/FMA3) 2/1 (AVX-512/FMA3) 2/1 (AVX2/FMA3)
L1L2 bandwidth 64 B/cy 64 B/cy 64 B/cy 32+32 B/cy
L2LLC bandwidth 32 B/cy 32 B/cy 16+16 B/cy 32 B/cy
LLC organization Inclusive Inclusive Exclusive Exclusive
Victim caches No No LLC LLC
El. transfers Non-overlapping Non-overlapping Non-overlapping Overlapping
```
```
Micro-architecture
```
```
Theor. MEM bandwidth 68 .3 GB/s 76 .8 GB/s 140 .8 GB/s 170 .6 GB/s
Compiler Intel C++v2019.5.281 Intel C++v2019.5.281 Intel C++v2019.5.281 Intel C++v2019.5.
Optimization flags -O3 -xHost -O3 -xHost -O3 -qopt-zmm-usage=high -O3 -xHost
(SIMD) -xAVX -xAVX -xCORE-AVX512 -mavx
Message passing library IntelMPIv2019u5 IntelMPIv2019u5 IntelMPIv2019u5 IntelMPIv2019u
SoftwareOperating system CentOS Linux v7.7.1908 Ubuntu 18.04.3 Ubuntu 18.04.3 Ubuntu 18.04.
ITAC v2019u4 v2019u4 v2019u4 v2019u
ToolsLIKWID v5.0.1 v5.0.1 v5.0.1 v5.0.
```
### nIIt

### n nIt

```
t
```
### nt

```
Fig. 4: Parameter space cov-
ered w.r.t. number of cores
sharing a bottleneck (dots).
nIt,nIIt: number of cores run-
ning kernelIandII;nt: num-
ber of cores on ccNUMA do-
main. Orange: domain fully
occupied; blue: symmetrical
scaling.
```
properties across systems, the modeling error is below 8%
globally and below 5% for 75% of cases.
Figure 4 visualizes the parameter space covered in this work
with respect to numbers of threads sharing a contention domain.
We only combine two distinct kernels, either filling the domain
completely (orange dots) or scaling with equal core counts
towards full saturation (blue dots). Other combinations were
ignored for brevity although they can be described by the
model just as well.

##### II. HARDWARE AND SOFTWARE SETUP,EXPERIMENTAL

METHODOLOGY
To ensure the broad applicability of our results, we conducted
all experiments on one ccNUMA domain of four different
processors from Intel and AMD (see Table I for details on
the hardware and software setup). Cluster-on-Die (CoD) and
Sub-NUMA Clustering (SNC) were turned off on Intel CPUs
(except for the experiments with HPCG on BDW in Sect.I-A),
NPS4 mode (i.e., four ccNUMA domains per socket) was
active on AMD Rome, and SMT threads were ignored, i.e.,
each application thread was assigned its own physical core. We
always employed the widest SIMD instruction set supported by
the architecture (AVX2 except on CLX where it was AVX-512)

```
and made sure that the compiler used standard (non-streaming)
stores. The clock speed (core and uncore) was fixed to the
base value via thelikwid-setFrequenciestool from
the LIKWID [9, 5] tool suite.
The transparent huge pages (THP) setting was put to “always,”
and the NUMA balancing feature was turned off in order to
reduce the performance impact from these settings [4]. All
prefetching mechanisms in the hardware were enabled.
In order to run two groups of threads of different size and
control thread-core affinity we used thelikwid-mpirun
tool. Working sets were chosen large enough to not fit into
any cache (i.e., at least 10×the last-level cache size), and data
sharing across overlapping kernels was ruled out to eliminate
the possibility of cache reuse.
Runtime traces were visualized using Intel trace analyzer
and collector (ITAC)^2. All timings were taken using the C++
high-resolutionChronoclock. Individual kernel executions
were repeated at least 15 times to even out variations in runtime.
Memory bandwidths were measured by taking the ratio of data
volume and wall-clock time.
The HPCG benchmark results presented in the in-
troduction were obtained with version 3.1, a problem
size of 1603 (2.7GB per rank), and a configured run-
time of 1800 seconds. In addition to the standard
compiler flags, we used -DHPCG_NO_OPENMP for pure
MPI,-DHPCG_CONTIGUOUS_ARRAYSfor contiguous data
layout, and -trace -tcollect -tcollect-filter
func.txt -qopt-reportto enable trace data collec-
tion and analysis of selected user functions (configured in
func.txt).
Further details can be obtained from the example code we
put up for download at http://tiny.cc/ISPASS-OBS.
```
(^2) https://software.intel.com/en-us/trace-analyzer


TABLE II: Key specifications of computational kernels used in the experiments. All arrays and scalars hold double-precision
floating point data. In all cases with write access, standard stores and write-back caches are assumed so that write-allocate
transfers cannot be avoided. Note that data transfers and code balance pertain to the L3 cache for the stencil codes.

```
Naive Pseudo-code for Elem. transf.Code balance Memory request fractionf Saturated bandw.bs[GB/s]
CasesKernels loop body (R+W+RFO)
```
```
∗ Bc[B/F] BDW-1BDW-2CLXRomeBDW-1BDW-2CLXRome
```
```
vectorSUM s += a[i] 1 (1+0+0) 8 0.241 0.178 0.1250.590 59 66.9 111.1 34.
DDOT1 s += a[i]*a[i] 1 (1+0+0) 4 0.242 0.179 0.1260.571 59 66.7 110.5 34.
DDOT2 s += a[i]*b[i] 2 (2+0+0) 8 0.252 0.181 0.1420.665 56.5 65.8 108.7 33.
Read-onlyDDOT3 s += a[i]*b[i]*c[i] 3 (3+0+0)^8 0.255 0.181 0.1660.721 56.8 65.5 100.9 33.
DSCAL a[i] = s * a[i] 2 (1+1+0) 16 0.374 0.301 0.2110.857 49.6 54.1 101.1 34.
DAXPY a[i] = a[i] + s * b[i] 3 (2+1+0) 12 0.315 0.239 0.2040.960 53.2 60.8 102.5 32.
ADD a[i] = b[i] + c[i] 4 (2+1+1) 32 0.309 0.228 0.1990.831 53.1 62.2 102 32.
STREAM a[i] = b[i] + s * c[i] 4 (2+1+1) 16 0.309 0.228 0.1990.838 53.2 62.2 102.4 32.
WAXPBY a[i] = r * b[i] + s * c[i] 4 (2+1+1) 10.67 0.309 0.228 0.1990.842 53.2 62.2 102.4 32.
DCOPY a[i] = b[i] 3 (1+1+1) 24 [B/row] 0.320 0.242 0.1900.803 53.5 60.9 104.2 32.
Schoenauer a[i] = b[i] + c[i] * d[i] 5 (3+1+1) 20 0.299 0.223 0.1850.859 53.1 60.5 101.7 31.
Jacobi-v1 §§
LCL2 † 3 (1+1+1) 6 0.252 0.195 0.1570.749 53.6 60.9 104.1 32.
LCL3 ‡ 5 (3+1+1) 10 0.141 0.104 0.1000.542 53.2 60.5 103.2 32.
Jacobi-v2 ¶
LCL2 † 4 (2+1+1) 2.46 0.247 0.188 0.1670.804 53.5 62.3 102.9 33.
```
```
Read-write
```
##### LCL3 ‡ 6 (4+1+1) 3.69 0.142 0.105 0.0880.458 52.9 60.8 103.2 32.

```
∗R: no. of read streams, W: no. of write streams, RFO: read-for-ownership (a.k.a. write allocate).
§§ Simple 2d 5-point stencil update: b[j][i] = (a[j][i-1] + a[j][i+1] + a[j-1][i] + a[j+1][i]) * s
† CL transfers and code balance in L3 for layer condition fulfilled at L2 (three data streams); grid size 20000×4000 (outer×inner)
‡ CL transfers and code balance in L3 for violated layer condition at L2 (five data streams); grid size 5000×25000 (outer×inner)
¶ More complicated 2d 5-point stencil update:
r1 = ( ax * (A[j][i-1] + A[j][i+1]) + ay * (A[j-1][i] + A[j+1][i]) + b1 * A[j][i] - F[j][i]) / b
B[j][i] = A[j][i] - relax * r
residual += r1 * r
```
##### III. CODE AND PERFORMANCE METRICS

```
Thecomputational intensity of a loop is the ratio of
arithmetic operations (“true work,” e.g.,flops) to the number of
bytes that must be transferred over a data bottleneck in order to
do the work. It is the main parameter that goes into the Roofline
model [10], which assumes only two hardware bottlenecks:
memory bandwidth and peak computational performance. Its
inverse, the code balance, appears at first sight to be a
reasonable metric to quantify the “hunger” of a loop for data.
However, from the point of view of refined performance models
such as the ECM model [8, 7], arithmetic instructions are an
overlapping component of code execution. This means that
the execution time of an executing loop does not depend on
the actual number offlopsbut on the time spent moving
data through the memory hierarchy unless the arithmetic work
becomes dominant. Theperformanceof the loop inflop/sdoes
depend on the number offlops, however, but this is not the
observable metric we are concerned with here.
We cannot give a full introduction to the ECM model here;
instead, we describe the components that are relevant for our
analysis on systems where the memory bandwidth is the sole
data transfer bottleneck on a ccNUMA domain. The following
contributions go into a single-core ECM runtime prediction:
```
- TOL: In-core execution time of loop instructions that are
    not contributing toTL 1 Reg
       - TL 1 Reg: Optimal execution time of all load and/or store
          instructions, assuming full pipeline throughput; i.e., mini-
          mum number of cycles required to retire these instructions
       - TMemTime for all required data transfers over the mem-
          ory interface assuming that the full saturated memory
          bandwidth can be utilized
       - {Ti}Times for data transfers over all relevant data paths in
          the cache hierarchy, i.e., L2-L3 and L1-L2 in a three-level
          cache hierarchy

```
The ECMmachine modelmakes assumptions about how these
contributions have to be put together to arrive at a single-core
runtime contribution. For example, on all Intel server CPUs to
date, only loads count towardsTL 1 Reg, and the runtime is
```
```
TECM=max
```
##### (

```
TOL,TMem+
```
```
n
```
## ∑

```
i= 1
```
```
Ti+TL 1 Reg
```
##### )

##### , (1)

```
i.e., data transfers are nonoverlapping while all instructions
executed (except loads) in the core overlap with everything. The
ECMapplication modelprovides information about the actual
code (instructions and their dependencies) and data transfer
volumes. It is usually obtained by static code analysis and
assumptions about data transfers. In case of pure streaming
loop kernels without temporal reuse, it is straightforward to
calculate the data volumes. For stencils, a layer condition
(LC) analysis [8] provides this information. The ECM model
```

```
Fig. 5: Basic model of memory bandwidth
sharing. In the example, two kernels gener-
ate requests on six and four cores, respec-
tively, and kernel II (on MPI rank 1) has
a much higherfIIfI. It can queue more
requests per core and thus get more share
of bandwidth per core.
```
```
RRR
Memory bus
```
```
Rank 0
```
```
Rank 1
```
```
C
C
C
C
C
C
C
C
C
C9 R: Request packet
```
```
b( 6 , 4 )·^6 ·f
I
6 ·fI+ 4 ·fII
```
```
b( 6 , 4 )·^4 ·f
II
6 ·fI+ 4 ·fII
```
```
Enqueue Dequeue
```
```
Back Front
```
```
FIFO Queue
```
```
RRR RRR RRR
RRR RRR RRR
RRR RRR RRR
RRR RRR RRR
RRR RRR RRR
RRR RRR RRR
RRR RRR
RRR RRR
RRR RRR
RRR RRR
```
```
provides us with a resource-centric point of view on memory
bandwidth utilization. The fraction of time the serial code
occupies the memory interface is
```
```
f=
```
```
TMem
TECM
```
##### . (2)

```
On a largely nonoverlapping memory hierarchy like in Intel
server CPUs, thememory request fraction f is significantly
smaller than one even for streaming kernels with no temporal
locality; on AMD Rome, which has strongly overlapping
characteristics, it is often close to one in this case. The value
offquantifies how much of the shared resource (the memory
interface) can be utilized by a single core. In most memory-
bound loops,fdoes not change if the number offlops(and
hence the code balance) changes because the data transfers
dominate in (1). This is why code balance is not a good metric
for quantifying the need of a loop for data. In contrast tof,
it does not take machine characteristics like the overlapping
behavior of the cache hierarchy into account.
The value off can be determined by using performance
counters to measure the memory bandwidth drawn when
executing a loop kernel in single-threaded mode. It can also be
predicted by the analytic ECM model if the required knowledge
about the hardware and code (see above) is available. We use
the first option here and determinefper kernel as
```
```
f=
```
```
bmeas
bs
```
##### , (3)

wherebmeasis the measured single-threaded memory bandwidth
andbsis the saturated (full-domain) memory bandwidth of the
kernel.
The ECM model also predicts the scaling behavior of loops
across cores on a contention domain. We use a simplified
version of the recursive model presented in [6]. It assumes that,
atncores, a latency penalty ofp 0 ×u(n− 1 )×(n− 1 )must be
added, withu(i)being the utilization of the memory interface
aticores,u( 1 ) =f, andp 0 =TMem/ 2. This is not as accurate
as the full model, in whichp 0 is a fit parameter, but it will
suffice for our purposes.
The maximum memory bandwidthbsis not the same for
all loop kernels. As a general rule on all x86 CPUs, read-
only kernels achieve a somewhat (5%–15%) higher saturated
bandwidth than kernels with write streams. While this effect is
of minor importance in single-core modeling since the memory
data transfer only accounts for a fraction (f) of the execution
time at least on Intel CPUs, an accurate model of bandwidth
sharing requires to take the differences inbsinto account.
Table II lists the loop kernels used for our experiments
together with their basic properties such as number of elements

```
transferred over the memory interface per iteration (except for
the stencils, where we show transfers to/from the L3 cache),
their (memory or L3) code balance, theirfvalues, and their
saturated bandwidths. We include two 2d stencil algorithms
with different characteristics. For each we select two different
grid sizes that exhibit layer conditions (LC) fulfilled or broken
at the L2 cache; if the LC is fulfilled, the data transfers between
L3 and L2 are reduced because reuse across the outer stencil
dimension is possible at the L2 cache. This happens when three
consecutive rows of the source grid fit into L2. For the stencils
we thus list the code balance in L3 instead of in memory,
because the in-memory code balance is the same regardless of
the LC at L2. However, the intra-cache data transfers between
L2 and L3 make a significant difference for the memory request
fractionf. See [8] for an in-depth coverage of layer conditions.
```
##### IV. ANALYTICAL MODEL

```
The central quantities in our bandwidth sharing model are the
kernel’s memory request fractionfand the saturated bandwidth
bs. As an example we look at a ten-core contention domain
with six cores running kernel I (fI) and four cores running
kernel II (fII) as depicted in Fig. 5. Each core issues requests
at a certain fraction of the maximum rate; this fraction isfI
orfII, respectively. In the model, this is also the rate at which
requests are queued to be serviced by the memory interface. A
kernel with higherfwill be able to queue more requests. Since
we are dealing with a fully populated contention domain, the
requests from the ten cores compete for bandwidth, which is
limited by a value that depends (though weakly) on the code
characteristics. This variation is phenomenological input to the
model. In Table II we listed the saturation bandwidthsbsfor
all kernels at homogeneous execution (no mixing). When two
different kernels overlap, this value changes, and we assume
that the overlapped saturated bandwidth is a weighted mean
of homogeneous bandwidths:
```
```
b(nIt,nIIt) =
```
```
nIt×bIs+nIIt×bIIs
nIt+nIIt
```
##### (4)

```
Here,bIsandbIIsare the saturated bandwidths for the two kernels
in homogeneous execution, andnItandnIIt are the respective
numbers of threads, wherent=nIt+nIIt.
As shown in Fig. 5, where we chosenIt= 6 andnIIt = 4 , we
now assume that group I of cores gets a share of all requests,
and thus a share of the bandwidth, proportional to
```
```
αI=
```
```
nIt×fI
nIt×fI+nIIt×fII
```
```
and αII= 1 −αI. (5)
```

```
20
```
```
40
```
```
60
DDOT
DCOPY
bDCOPY
```
(^20) bDCOPYs +DDOT 2
40
70
40
80
120
20
40
2
4
6
8
DCOPY
Model DCOPY
DDOT
[GB/s] Model DDOT
2
4
6
8
2
4
6
8
2
4
6
8
2
4
6
8
JacobiLCL3-v
Model JacobiLCL3-v
DDOT
Bandwidth Model DDOT
2
4
6
8
2
4
6
8
2
4
6
8
0 1 2 3 45 6 7 8 910
5
5. 2
5. 4
5. 6
5. 8
6
Thread count
STREAM
Model STREAM
JacobiLCL2-v
Model JacobiLCL2-v
109 8 7 65 4 3 2 1 0
(a) BDW-
0 2 4 6 81012141618
2. 5
3
3. 5
4
Thread count
18161412108 6 4 2 0
(b) BDW-
0 2 4 6 8101214161820
4
4. 5
5
5. 5
6
Thread count
20181614121086 4 2 0
(c) CLX
0 1 2 3 4 5 6 7 8
3. 6
3. 8
4
4. 2
4. 4
Thread count
8 7 6 5 4 3 2 1 0
(d) Rome
Fig. 6: Observed bandwidth per kernel vs. number of threads per kernel for the fully populated ccNUMA domain (thread
parameter space: orange dots in Fig. 4) across all four architectures (columns (a)–(d)). Top row: Stacked graph of bandwidth
share per kernel (DCOPY+DDOT2); the sum is the overall memory bandwidth (blue line). Second row: comparison of model
(lines) with measurement (symbols) for the per-core memory bandwidth using the same pairing as above. Third and fourth row:
same comparison for JacobiL3-v1+DDOT1 and STREAM+JacobiL2-v1, respectively.
In the homogeneous casefI=fII, the share of bandwidth is
solely determined by the number of threads in each group, but
different request fractions modify this simple linear behavior.
The bandwidth obtained by group I is thusαI×b(nIt,nIIt).
This model, although we have derived it starting from the
assumption that the ccNUMA domain is fully populated, can
also be applied to the nonsaturated case. In the following
section we will validate the model using measurements in the
parameter space depicted in Fig. 4.

##### V. MODEL VALIDATION AND DISCUSSION

In this section we first present measurements of selected
scenarios (i.e., kernel pairings) on the four architectures and
compare with the model (5), distinguishing between the fully
populated domain case and the case where the number of
cores goes up till saturation. We then extend the view to more
pairings and finally present an overview of the modeling error
for all cases.
In Fig. 6 we show three different kernel pairing scenarios per
architecture using fully populated ccNUMA domains (i.e., 10,
18, 20, and 8 cores on the four architectures (a)–(d)), covering
the orange dots in Fig. 4. The top panel in each column shows
a stacked graph of the bandwidth share of kernel I (DCOPY,
green) versus kernel II (DDOT2, hatched red) as the number of
threads on kernel I is increased and the number of threads on
kernel II is reduced. The top line in the graph is the measured
overall memory bandwidth. Since DCOPY has a higherf
than DDOT2 (see Table II), we expect from the model (5)
that DCOPY will get a higher share of the bandwidth as the
number of DCOPY threads goes up. This is observed as the
upward “bend” of the separator between the two regions of
the graph. The overall memory bandwidth goes down because

```
the saturation bandwidth of DCOPY is smaller than that of
DDOT2 (which is a read-only kernel that gets a little more
bandwidth out of the memory interface). This behavior is quite
universal across all four architectures for this kernel pairing.
The second panel in each column is a direct comparison
between the modeled (lines) and observed (symbols) memory
bandwidth per core for the DCOPY+DDOT2 case. Since
we chose a zoomed-inyaxis, the general downward trend
stemming from the decline in saturated bandwidth is now
more pronounced. Although not directly visible from the
mathematical description, this change is just as important for
the observed bandwidth as the difference inf. Obviously, the
model describes the per-core bandwidth quite accurately.
The third and fourth panels in each column show per-
core bandwidth for two other pairings: JacobiL3-v1+DDOT
and STREAM+JacobiL2-v1. Especially in the latter case the
saturated bandwidths of the two kernels are very similar, so the
downward trend is weaker and also the bandwidth difference
per core is smaller. Nevertheless the model also provides an
accurate prediction here.
Figure 7 shows the same cases as above but for symmetrical
thread scaling, running the same number of threads per kernel
(blue dots in Fig. 4). There is good agreement between model
and measurement along the bandwidth saturation curve as
well, but some peculiarities are worth noting. The notable
breakdown in bandwidth for DCOPY near saturation on BDW-
1 is reminiscent of the overall scaling behavior of this kernel on
this architecture, which shows a maximum bandwidth already
before the domain is full. While the BDW CPUs and especially
Rome show a rather strong decline in per-thread bandwidth
already at two threads per kernel, CLX scales well from two to
four, which reflects the fact that its single-core bandwidth is low
```

```
20
```
```
40
```
```
60
DDOT
DCOPY
bDCOPY
```
(^20) bDCOPYs +DDOT 2
40
70
40
80
120
10
20
30
40
4
6
8
10
12
DCOPY
Model DCOPY
DDOT
[GB/s]^5 Model DDOT
10
5
10
15
20
5
10
15
20
4
6
8
10
12
Thread count
JacobiLCL3-v
JacobiLCL3-v
DDOT
Bandwidth Model DDOT
1 2 3 4 5 6 7 8 9
4
6
8
10
12
Thread count
1 2 3 4 5 6 7 8 910
5
10
15
20
Thread count
1 2 3 4
5
10
15
20
Thread count
1 2 3 4 5
4
6
8
10
12
14
Thread count
STREAM
Model STREAM
JacobiLCL2-v
Model JacobiLCL2-v
1 2 3 4 5
(a) BDW-
1 2 3 4 5 6 7 8 9
4
6
8
10
12
Thread count
1 2 3 4 5 6 7 8 9
(b) BDW-
1 2 3 4 5 6 7 8 910
5
10
15
20
Thread count
1 2 3 4 5 6 7 8 910
(c) CLX
1 2 3 4
5
10
15
20
Thread count
1 2 3 4
(d) Rome
Fig. 7: Observed memory bandwidth per kernel vs. number of threads in symmetrical configuration, scaling across the ccNUMA
domain (thread parameter space: blue dots in Fig. 4) for all architectures (columns (a)–(d)). Organization, kernels, and data
plotted as in Fig. 6.
compared to its saturated bandwidth. Rome is special in terms
of scaling since all kernels can almost saturate the memory
bandwidth already with one thread due to its overlapping cache
hierarchy. A more complete coverage of scaling results for
more pairings can be found at [http://tiny.cc/ISPASS-OBS.](http://tiny.cc/ISPASS-OBS.)
In Fig. 8 we give a concise overview of the relative modeling
error in per-core bandwidth across 30 kernel pairings along
the bandwidth scaling curve, i.e., for symmetrical pairing. In
75% of all cases the error is below 5%, and the maximum
overall error is 8%. In view of the significant differences in
architectural details across the four CPUs (inclusive vs. victim
LLC, shared vs. segmented LLC, overlapping vs. serializing
caches) we consider this an exceptional result.
Figure 9 shows an overview of 32 kernel pairings (includ-
ing self pairings) of vecSUM, DDOT2, DDOT3, DCOPY,
Schoenauer, DAXPY, DSCAL, JacobiL2-v1, JacobiL3-v1, and
TRIAD with others using an equal share of threads on a full
contention domain. Each row of bars (a)–(d) refers to one
architecture. Each bar represents the relative bandwidth share of
the first kernel in the pair (e.g., vecSUM in vecSUM+DDOT2)
as opposed to the homogeneous situation (e.g., vecSUM paired
with itself). All bars in a group are thus normalized to the first
bar.
The first observation is that the patterns observed in each
group of bars are quite consistent across architectures: Whether
bandwidth is gained or lost with respect to the homogeneous
situation depends on the ratio offvalues of the kernels being
greater or smaller than one. Across the Intel CPUs, this criterion
is independent of the architecture. The CLX CPU is still special
in the sense that the differences in bandwidth are smaller overall.
One might be tempted to attribute this to the fact that it needs
more cores to saturate the memory bandwidth than the BDW
variants, i.e., it is “more scalable.” In other words, the memory
transfers need less time relative to the other contributions in
the single-core ECM model, which cause generally smaller
request fractionsfas can be seen in Table II. However, a global
reduction factor infcancels out in the model (5), hence this
cannot be the reason for the weaker bandwidth variations. The
1 2 3 4 5
0
2
4
6
8
10
Thread count
Percentage error

##### [%

##### ]

```
1 2 3 4 5
```
```
(a) BDW-
```
```
1 2 3 4 5 6 7 8 9
0
```
```
2
```
```
4
```
```
6
```
```
8
```
```
10
```
```
Thread count
```
```
1 2 3 4 5 6 7 8 9
```
```
(b) BDW-
```
```
1 2 3 4 5 6 7 8 910
0
```
```
2
```
```
4
```
```
6
```
```
8
```
```
10
```
```
Thread count
```
```
1 2 3 4 5 6 7 8 910
```
```
(c) CLX
```
```
1 2 3 4
0
```
```
2
```
```
4
```
```
6
```
```
8
```
```
10
```
```
Thread count
```
```
1 2 3 4
```
```
(d) Rome
```
```
Fig. 8: Overview of modeling error across architectures (a)–(d) for symmetrical thread scaling (blue dots in Fig. 4) and 30
pairings per thread count and architecture. The error is calculated as|(bObserved−bModel)/bModel|. Whiskers denote min/max
error, boxes encompass second and third quartiles, and the median is marked.
```

```
Normalized runtime
```
0. (^6) (a) BDW-15:5 cores
0. 8
1
1. 2
1. 4
0. 6
0. 8
1
1. 2
1. 4
(b) BDW-29:9 cores
0. 6
0. 8
1
1. 2
1. 4
(c) CLX10:10 cores
vecSUM+vecSUMvecSUM+DDOT2vecSUM+DCOPY
vecSUM+Schoenauer
DDOT+DDOT2DDOT2+vecSUMDDOT2+DCOPY
DDOT2+Schoenauer
DDOT3+DDOT3DDOT3+DCOPY
DDOT3+Schoenauer
DCOPY+DCOPYDCOPY+vecSUMDCOPY+DDOT2DCOPY+DDOT
Schoenauer+Schoenauer
Schoenauer+vecSUMSchoenauer+DDOT2Schoenauer+DDOT
DAXPY+DAXPYDAXPY+DSCALDSCAL+DSCALDSCAL+DAXPY
Jacobi+Jacobi
LCL
-v
Jacobi
LCL
-v1+TRIAD
Jacobi+Jacobi
LCL
-v
Jacobi
LCL
-v1+TRIAD
Jacobi
LCL
-v1+DDOT
TRIAD+TRIAD
TRIAD+Jacobi
LCL
-v
TRIAD+Jacobi
LCL
-v
TRIAD+Jacobi
LCL
-v
0. 6
0. 8
1
1. 2
1. 4
(d) Rome4:4 cores
Fig. 9: Bandwidth gain or loss for symmetrical kernel pairings (each kernel gets half the contention domain) across architectures
(a)–(d). The height of each bar is the relative gain or loss in bandwidth of the first kernel in the pair when paired with the
second, normalized to the self-paired case (first bar in each group).
real reason for CLX showing smaller variations is two-fold:
first, it shows less spread in saturated bandwidth than the other
Intel CPUs across all kernels in Table II (10% as opposed
to 20% for BDW-1); second, it also shows less spread inf
values (2.4 vs. 2.7 for BDW-1). These two factors together
lead to the reduced sensitivity of shared bandwidth to kernel
variations on CLX. The AMD Rome CPU shows different
patterns from the Intel chips for some combinations, especially
for DAXPY+DSCAL. This due tofDAXPY>fDSCALon Rome
as opposed to the Intel CPUs where this relation is reversed.
We started this discussion with a close look at plain and
modified MPI-parallel HPCG variants (see Sect.I-A), where
we observed how back-to-back compute kernels overlapping
on a contention domain due to desynchronization could either
lead to them slowing down or speeding up depending on the
particular pairing. If a kernel is sandwiched between a high-
fkernel coming before it and a low-f kernel coming after
it, early starters get slowed down and late starters get sped
up. This could be observed in Fig. 3(b), where the follow-
up kernel to DDOT2 was DAXPY, withfDAXPY= 0. 315 and
fDDOT2= 0. 252. This large difference directly leads to the pos-
itive observed skewness and means that the desynchronization
is amplified in such a situation. Overlapping with idleness, as
shown in Fig. 3(a) on the trailing edge of the DDOT2 execution,
causes resynchronization. Our model is thus not only good for
a quantitative description of performance differences on shared
bottlenecks; it can also predict qualitatively the dynamics of
desynchronization and resynchronization in memory-bound
bulk-synchronous barrier-free programs.

##### VI. CONCLUSION AND OUTLOOK

```
Starting from observations of desynchronized kernel execu-
tion on the MPI-parallel HPCG benchmark we motivated the
need for an analytical model of bandwidth sharing between
groups of threads executing different loop kernels with different
characteristics on a bandwidth contention domain. Based on the
principles of the Execution-Cache-Memory (ECM) performance
model, we constructed and validated such a model on four
current x86 server processors from Intel and AMD. We could
show that the major influence factors for bandwidth sharing
between two groups of threads are each kernel’s memory
request frequency, which is directly related to its single-threaded
memory bandwidth, and its saturated bandwidth on the domain.
Across a variety of pairings of kernels with and without cache
reuse, the observed error in predicted per-core bandwidth was
never larger than 8%, and lower than 5% for 75% of all
cases. Apart from the quantification of bandwidth shares, the
model is also able to predict how desynchronization across MPI
processes gets amplified or mitigated depending on which back-
to-back kernels are overlapped with each other in MPI-parallel
barrier-free bulk-synchronous programs.
The set of loop kernels chosen here is a reasonable cross
section with a spectrum of properties. Since the memory request
fraction and the saturated memory bandwidth are the only
relevant parameters in our model, it should be applicable also
for more complex kernels (e.g., with more concurrent data
streams or with dominant in-core execution). It should also be
useful in modeling the performance of task-parallel code where
the synchronized, data-parallel execution of threads may be
```

more the exception than the rule. A validation of the model on
more processor architectures, e.g., Power- or Arm- based CPUs,
is certainly in order. Finally, our work enables the development
of a new kind of MPI simulation technique that can take
node-level bottlenecks into account much more accurately than
previously possible. We leave these investigations for future
work.

##### ACKNOWLEDGMENTS

```
This work was supported by KONWIHR, the Bavarian Com-
petence Network for Scientific High Performance Computing
in Bavaria, under project name “OMI4papps.”
```
##### REFERENCES

```
[1] A. Afzal, G. Hager, and G. Wellein. Delay flow mech-
anisms on clusters.URL: https://hpc.fau.de/files/2019/
09/EuroMPI2019_AHW-Poster.pdf. Poster at EuroMPI
2019, September 10–13, 2019, Zurich, Switzerland.
[2] A. Afzal, G. Hager, and G. Wellein. Desynchroniza-
tion and wave pattern formation in MPI-parallel and
hybrid memory-bound programs. In P. Sadayappan,
B. L. Chamberlain, G. Juckeland, and H. Ltaief, editors,
High Performance Computing, pages 391–411, Cham.
Springer International Publishing, 2020.ISBN: 978-3-
030-50743-5.DOI: 10.1007/978-3-030-50743-5_20.
[3] A. Afzal, G. Hager, and G. Wellein. Propagation and
decay of injected one-off delays on clusters: A case
study. In2019 IEEE International Conference on Cluster
Computing, CLUSTER 2019, Albuquerque, NM, USA,
September 23-26, 2019, pages 1–10, 2019.DOI: 10.1109/
CLUSTER.2019.8890995.
[4] C. L. Alappat et al. Understanding HPC benchmark
performance on Intel Broadwell and Cascade Lake
processors. In P. Sadayappan, B. L. Chamberlain, G.
Juckeland, and H. Ltaief, editors,High Performance
Computing, pages 412–433, Cham. Springer Interna-
tional Publishing, 2020.ISBN: 978-3-030-50743-5.DOI:
https://doi.org/10.1007/978-3-030-50743-5_21.
[5] J. Eitzinger and T. Röhl. LIKWID web pages. https :
//githib.com/RRZE-HPC/likwid.
[6] J. Hofmann, G. Hager, and D. Fey. On the accuracy
and usefulness of analytic energy models for contem-
porary multicore processors. In R. Yokota, M. Weiland,
D. Keyes, and C. Trinitis, editors,High Performance
Computing, pages 22–43, Cham. Springer International
Publishing, 2018.ISBN: 978-3-319-92040-5.DOI: 10.
1007/978-3-319-92040-5_2.
[7] J. Hofmann et al. Bridging the architecture gap: abstract-
ing performance-relevant properties of modern server
processors.Supercomputing Frontiers and Innovations,
7(2), 2020.ISSN: 2313-8734.DOI: 10.14529/jsfi200204.
[8] H. Stengel, J. Treibig, G. Hager, and G. Wellein.
Quantifying performance bottlenecks of stencil com-
putations using the Execution-Cache-Memory model. In
Proceedings of the 29th ACM International Conference
on Supercomputing, ICS ’15, Newport Beach, CA. ACM,
2015.DOI: 10.1145/2751205.2751240.
```
```
[9] J. Treibig, G. Hager, and G. Wellein. LIKWID: a
lightweight performance-oriented tool suite for x86 mul-
ticore environments. In2012 41st International Confer-
ence on Parallel Processing Workshops, pages 207–216,
Los Alamitos, CA, USA. IEEE Computer Society, Sept.
2010.DOI: 10.1109/ICPPW.2010.38.
[10] S. Williams, A. Waterman, and D. Patterson. Roofline:
an insightful visual performance model for multicore
architectures.Commun. ACM, 52(4):65–76, 2009.ISSN:
0001-0782.DOI: 10.1145/1498765.1498785.
```


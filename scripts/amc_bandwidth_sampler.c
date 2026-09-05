// ============================================================================
// Per-engine DRAM-bandwidth sampler (Apple Silicon, no root required)
//
// Streams per-engine DRAM traffic as CSV, one line per sampling interval, from
// the private IOReport framework. Two backends, chosen automatically at startup
// by what actually samples on this machine (never by chip name):
//
//   backend=0  "amc"  M2-family. "AMC Stats / Perf Counters / * DCS RD|WR":
//                     monotonic per-requestor BYTE counters. Exact.
//   backend=1  "pmp"  M3-family. "PMP / DCS BW / <REQ> RD|WR": 32-bin BANDWIDTH
//                     HISTOGRAMS (kIOReportFormatState). Bin i is labelled with
//                     its upper edge in GB/s; its residency is the number of PMP
//                     ticks spent in that bin. Bytes are derived, not counted.
//
// Why two: on M3-family silicon the AMC channels enumerate but
// AppleH15MemCacheController REFUSES the subscription, so they never reach a
// sample at all (they are not "present but zero" -- they are absent). See
// docs/amc-m3-counters-plan.md and scripts/ior_stage.c.
//
// Only "DCS" (DRAM-command-scheduler) channels are read on either backend --
// that is memory-controller-side truth. "PMP / AF BW / *" is the fabric-side
// equivalent and must NOT be used for byte claims.
//
// Requestors bucket as: PCPU*/ECPU*/PACC*/EACC* -> cpu, GFX*/AGX* -> gpu,
// ANE*/ANS* -> ane, everything else -> other.
//
// CSV columns:
//   timestamp    wall-clock seconds (matches the framework CSV's %(created)f,
//                used for cross-process trace alignment), taken at interval end
//   dt_s         measured interval length (monotonic clock)
//   {cpu,gpu,ane,other}_{rd,wr}  bytes moved during the interval
//   total_rd, total_wr           sums of the above
//   total_gbps   controller-aggregate GB/s
//   {cpu,gpu,ane,other}_duty     pmp: fraction of the interval the engine was
//                                powered (its counters only tick when it is);
//                                amc: always 1
//   backend      0 = amc byte counters, 1 = pmp histograms
//   saturated    1 if any channel put ticks in its TOP bin this interval, i.e.
//                it ran past the counter's range and the row is a LOWER BOUND
//
// Every column is numeric on purpose: the readers (staged_lib.load_bandwidth_csv,
// summarize()) float() every field and drop rows that fail, so a text column
// here would silently empty the CSV.
//
// pmp backend caveats -- read docs/amc-m3-counters-plan.md before quoting bytes:
//   * per-requestor bins cap at 32 GB/s, the aggregate at 64 GB/s -> `saturated`
//   * bin 0 is a "<1 GB/s" catch-all scored at its 0.5 GB/s midpoint, so a
//     powered-but-idle engine shows a small phantom floor (a gated one shows 0)
//   * a duty-cycled engine's bytes are estimated by normalising against the
//     never-gated aggregate channel's tick count; for smooth, sub-ceiling loads
//     (what the paper reports) this agrees with the naive mean to <1%
//
// Usage:  amc_bandwidth_sampler [-i interval_ms] [-n samples] [-o out.csv] [--raw]
//         -n 0 (default) = run until SIGINT/SIGTERM (lines are flushed as
//         written, so the file is valid whenever the process is killed).
//         --raw additionally prints every live channel per interval.
//
// Build (done automatically by amc_bandwidth_sampler.py):
//   clang -O2 -o amc_bandwidth_sampler amc_bandwidth_sampler.c -framework CoreFoundation
// ============================================================================
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef CFMutableDictionaryRef (*FnCopyAll)(uint64_t, uint64_t);
typedef CFMutableDictionaryRef (*FnCopyGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void* (*FnCreateSub)(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*FnCreateSamples)(void*, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*FnSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef int (*FnIterate)(CFDictionaryRef, int (^)(CFDictionaryRef));
typedef CFStringRef (*FnGetStr)(CFDictionaryRef);
typedef int64_t (*FnGetInt)(CFDictionaryRef, void*);  // 2nd arg ABI is unstable across headers; only NULL is safe
typedef int32_t (*FnStateCount)(CFDictionaryRef);
typedef int64_t (*FnStateRes)(CFDictionaryRef, int32_t);
typedef CFStringRef (*FnStateName)(CFDictionaryRef, int32_t);

static FnIterate Iterate;
static FnGetStr GetGroup, GetSubGroup, GetChannelName;
static FnGetInt GetInt;
static FnStateCount StateCount;
static FnStateRes StateRes;
static FnStateName StateName;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

static void cstr(CFStringRef s, char* buf, size_t n) {
    buf[0] = 0;
    if (s) CFStringGetCString(s, buf, n, kCFStringEncodingUTF8);
}
static double now_real(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}
static double now_mono(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

enum { B_CPU, B_GPU, B_ANE, B_OTHER, B_N };
enum { BACKEND_AMC = 0, BACKEND_PMP = 1 };

// Accumulators live at file scope: blocks capture statics by plain reference
// (no __block byref cell -- clang's byref support for aggregate types is what
// broke the first version of this loop).
static int64_t s_rd[B_N], s_wr[B_N];
static int64_t s_agg_rd, s_agg_wr;  // bare "DCS RD/WR" (no requestor) = MC aggregate
static double  p_wsum[B_N][2];      // pmp: GB/s-weighted ticks, [bucket][0=rd,1=wr]
static int64_t p_ticks[B_N];        // pmp: powered ticks for the bucket
static double  p_agg_wsum;
static int64_t p_agg_ticks;
static int     p_sat;
static FILE*   s_rawout;
static int     s_backend;

// Requestor token -> bucket. Returns -1 for a channel with no requestor (the
// memory-controller aggregate). M2 names channels "DIE0 GFX DCS RD"; M3 names
// the same channel "GFX DCS RD", so key on the REQUESTOR TOKEN, never on a
// "DIE" prefix -- doing the latter read every M3 per-requestor channel as the
// aggregate and left cpu/gpu/ane at zero.
static int bucket_of(const char* name) {
    if (strstr(name, "PCPU") || strstr(name, "ECPU") ||
        strstr(name, "PACC") || strstr(name, "EACC")) return B_CPU;
    if (strstr(name, "GFX")  || strstr(name, "AGX"))  return B_GPU;
    if (strstr(name, "ANE")  || strstr(name, "ANS"))  return B_ANE;
    if (strstr(name, "AVD")  || strstr(name, "ISP")  || strstr(name, "JPEG") ||
        strstr(name, "DISP") || strstr(name, "SIO")  || strstr(name, "PMP")  ||
        strstr(name, "AVE")  || strstr(name, "MSR")  || strstr(name, "SCODEC") ||
        strstr(name, "IOA")) return B_OTHER;
    return -1;
}

// Is this an "AMC Stats" per-byte DCS channel?
static int amc_dcs_op(const char* group, const char* name) {
    if (!strstr(group, "AMC")) return 0;
    if (strstr(name, "DCS RD")) return 1;
    if (strstr(name, "DCS WR")) return 2;
    return 0;
}

// Is this a "PMP / DCS BW" histogram channel? Returns 1 for RD, 2 for WR, 3 for
// the redundant RD+WR (which must never be summed into rd/wr), 0 otherwise.
// Names are "<REQ> RD", "<REQ> WR", "<REQ> RD+WR", or a bare "RD+WR" aggregate.
static int pmp_bw_op(const char* group, const char* subgroup, const char* name) {
    if (strcmp(group, "PMP") || strcmp(subgroup, "DCS BW")) return 0;
    const char* sp = strrchr(name, ' ');
    const char* op = sp ? sp + 1 : name;
    if (!strcmp(op, "RD")) return 1;
    if (!strcmp(op, "WR")) return 2;
    if (!strcmp(op, "RD+WR")) return 3;
    return 0;
}

// Mean bandwidth of one histogram channel: returns GB/s-weighted ticks in
// *wsum, tick count in *ticks, and sets *sat if the top bin took any ticks.
static void hist_stats(CFDictionaryRef ch, double* wsum, int64_t* ticks, int* sat) {
    *wsum = 0; *ticks = 0;
    int32_t c = StateCount ? StateCount(ch) : 0;
    if (c <= 0 || c > 64) return;
    for (int32_t i = 0; i < c; i++) {
        int64_t r = StateRes(ch, i);
        if (r <= 0) continue;
        char sn[128]; cstr(StateName(ch, i), sn, sizeof sn);
        double label = atof(sn);              // bin upper edge, GB/s
        if (label <= 0) continue;
        double width = label / (double)(i + 1);  // uniform bins
        *wsum += (double)r * (label - width / 2.0);
        *ticks += r;
        if (i == c - 1) *sat = 1;
    }
}

// Count the channels of each backend that are actually LIVE in a sample.
static int g_n_amc, g_n_pmp;
static void census(CFDictionaryRef sample) {
    g_n_amc = g_n_pmp = 0;
    Iterate(sample, ^int(CFDictionaryRef ch) {
        char g[128], sg[128], nm[256];
        cstr(GetGroup(ch), g, sizeof g);
        cstr(GetSubGroup(ch), sg, sizeof sg);
        cstr(GetChannelName(ch), nm, sizeof nm);
        if (amc_dcs_op(g, nm)) {
            // Monotonic byte counters: on a machine that populates them at
            // least one is non-zero in the very first sample.
            if (GetInt(ch, NULL) > 0) g_n_amc++;
        } else if (pmp_bw_op(g, sg, nm)) {
            double w; int64_t t; int s = 0;
            hist_stats(ch, &w, &t, &s);
            if (t > 0) g_n_pmp++;
        }
        return 0;
    });
}

int main(int argc, char** argv) {
    long interval_ms = 500, max_samples = 0;
    const char* out_path = NULL;
    int raw = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) interval_ms = atol(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_samples = atol(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else { fprintf(stderr, "usage: %s [-i ms] [-n samples] [-o file] [--raw]\n", argv[0]); return 2; }
    }
    if (interval_ms < 50) { fprintf(stderr, "interval < 50 ms refused (sampling overhead)\n"); return 2; }

    void* h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen libIOReport failed: %s\n", dlerror()); return 1; }
    FnCopyAll CopyAll = (FnCopyAll)dlsym(h, "IOReportCopyAllChannels");
    FnCopyGroup CopyGroup = (FnCopyGroup)dlsym(h, "IOReportCopyChannelsInGroup");
    FnCreateSub CreateSub = (FnCreateSub)dlsym(h, "IOReportCreateSubscription");
    FnCreateSamples CreateSamples = (FnCreateSamples)dlsym(h, "IOReportCreateSamples");
    FnSamplesDelta SamplesDelta = (FnSamplesDelta)dlsym(h, "IOReportCreateSamplesDelta");
    Iterate = (FnIterate)dlsym(h, "IOReportIterate");
    GetGroup = (FnGetStr)dlsym(h, "IOReportChannelGetGroup");
    GetSubGroup = (FnGetStr)dlsym(h, "IOReportChannelGetSubGroup");
    GetChannelName = (FnGetStr)dlsym(h, "IOReportChannelGetChannelName");
    GetInt = (FnGetInt)dlsym(h, "IOReportSimpleGetIntegerValue");
    StateCount = (FnStateCount)dlsym(h, "IOReportStateGetCount");
    StateRes = (FnStateRes)dlsym(h, "IOReportStateGetResidency");
    StateName = (FnStateName)dlsym(h, "IOReportStateGetNameForIndex");
    if (!CreateSub || !CreateSamples || !SamplesDelta || !Iterate || !GetGroup ||
        !GetSubGroup || !GetChannelName || !GetInt || !StateCount || !StateRes || !StateName) {
        fprintf(stderr, "dlsym failed (IOReport ABI changed?)\n"); return 1;
    }

    // Backend selection by what actually SAMPLES, not by chip name.
    //
    // 1. Try the cheap AMC-group-scoped subscription (the M2 path: ~117
    //    channels rather than ~9000).
    // 2. Otherwise subscribe to every channel and look again -- the M3's
    //    group-scoped copy is rejected outright by the memory-controller
    //    driver, and its AMC channels are silently dropped from the sample even
    //    via the all-channel path, so this is also where PMP is found.
    CFMutableDictionaryRef chans = NULL, sub_chans = NULL;
    void* sub = NULL;
    CFDictionaryRef probe = NULL;

    if (CopyGroup) {
        chans = CopyGroup(CFSTR("AMC Stats"), NULL, 0, 0, 0);
        if (chans) {
            sub = CreateSub(NULL, chans, &sub_chans, 0, NULL);
            if (sub) {
                probe = CreateSamples(sub, sub_chans, NULL);
                census(probe);
                if (g_n_amc > 0) s_backend = BACKEND_AMC;
                else { CFRelease(probe); probe = NULL; sub = NULL; }
            }
        }
    }
    if (!sub) {
        if (!CopyAll) { fprintf(stderr, "IOReportCopyAllChannels unavailable\n"); return 1; }
        chans = CopyAll(0, 0);
        if (!chans) { fprintf(stderr, "IOReport channel copy failed\n"); return 1; }
        sub = CreateSub(NULL, chans, &sub_chans, 0, NULL);
        if (!sub) { fprintf(stderr, "IOReportCreateSubscription failed\n"); return 1; }
        probe = CreateSamples(sub, sub_chans, NULL);
        census(probe);
        if (g_n_amc > 0) s_backend = BACKEND_AMC;
        else if (g_n_pmp > 0) s_backend = BACKEND_PMP;
        else {
            fprintf(stderr,
                "amc_bandwidth_sampler: no per-engine DRAM counters are readable on this "
                "machine. Neither 'AMC Stats / * DCS RD|WR' (M2-family byte counters) nor "
                "'PMP / DCS BW / *' (M3-family histograms) yielded a live channel in a "
                "sample. Refusing to emit an all-zero trace. Run "
                "scripts/preflight_bandwidth_counters.sh; see docs/amc-m3-counters-plan.md.\n");
            return 3;
        }
    }

    FILE* out = stdout;
    if (out_path) {
        out = fopen(out_path, "w");
        if (!out) { perror("fopen"); return 1; }
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    fprintf(out, "timestamp,dt_s,cpu_rd,cpu_wr,gpu_rd,gpu_wr,ane_rd,ane_wr,"
                 "other_rd,other_wr,total_rd,total_wr,total_gbps,"
                 "cpu_duty,gpu_duty,ane_duty,other_duty,backend,saturated\n");
    fflush(out);
    fprintf(stderr, "amc_bandwidth_sampler: backend=%s (%d live channels), interval %ld ms, %s\n",
            s_backend == BACKEND_AMC ? "amc" : "pmp",
            s_backend == BACKEND_AMC ? g_n_amc : g_n_pmp,
            interval_ms, out_path ? out_path : "stdout");
    if (s_backend == BACKEND_PMP)
        fprintf(stderr, "amc_bandwidth_sampler: pmp histograms -- bytes are DERIVED; "
                        "per-requestor range caps at 32 GB/s (watch the 'saturated' column) "
                        "and sub-1 GB/s traffic is indistinguishable from idle.\n");

    CFDictionaryRef prev = probe ? probe : CreateSamples(sub, sub_chans, NULL);
    double t_prev = now_mono();
    struct timespec sleep_ts = { interval_ms / 1000, (interval_ms % 1000) * 1000000L };

    for (long k = 0; (max_samples == 0 || k < max_samples) && !g_stop; k++) {
        nanosleep(&sleep_ts, NULL);
        if (g_stop) break;
        CFDictionaryRef cur = CreateSamples(sub, sub_chans, NULL);
        double t_cur = now_mono();
        CFDictionaryRef delta = SamplesDelta(prev, cur, NULL);
        double dt = t_cur - t_prev, wall = now_real();

        memset(s_rd, 0, sizeof s_rd);
        memset(s_wr, 0, sizeof s_wr);
        memset(p_wsum, 0, sizeof p_wsum);
        memset(p_ticks, 0, sizeof p_ticks);
        s_agg_rd = s_agg_wr = 0;
        p_agg_wsum = 0; p_agg_ticks = 0; p_sat = 0;
        s_rawout = raw ? out : NULL;

        Iterate(delta, ^int(CFDictionaryRef ch) {
            char g[128], sg[128], name[256];
            cstr(GetGroup(ch), g, sizeof g);
            cstr(GetSubGroup(ch), sg, sizeof sg);
            cstr(GetChannelName(ch), name, sizeof name);

            if (s_backend == BACKEND_AMC) {
                int op = amc_dcs_op(g, name);
                if (!op) return 0;
                int64_t v = GetInt(ch, NULL);
                if (v < 0) return 0;  // counter wrap / bogus sample
                // The bare "DCS RD/WR" channel (no requestor token) is the
                // memory-controller AGGREGATE = the true total. Summing it AND
                // its per-requestor components double-counts (the calibration
                // bug that once reported 2x the real total).
                int b = bucket_of(name);
                if (b < 0) { if (op == 1) s_agg_rd += v; else s_agg_wr += v; }
                else       { if (op == 1) s_rd[b] += v; else s_wr[b] += v; }
                if (s_rawout && v > 0) fprintf(s_rawout, "# %s = %lld\n", name, (long long)v);
            } else {
                int op = pmp_bw_op(g, sg, name);
                if (!op) return 0;
                double w; int64_t t; int sat = 0;
                hist_stats(ch, &w, &t, &sat);
                if (t <= 0) return 0;
                int b = bucket_of(name);
                if (b < 0) {
                    // Bare "RD+WR" (M3) or "AMCC RD+WR" (M2) = controller
                    // aggregate. It is never power-gated, so its tick count is
                    // the interval's time base for every other channel.
                    if (op == 3) { p_agg_wsum = w; p_agg_ticks = t; if (sat) p_sat = 1; }
                    return 0;
                }
                if (op == 3) return 0;  // redundant with RD and WR; never sum it
                p_wsum[b][op - 1] += w;
                if (t > p_ticks[b]) p_ticks[b] = t;
                if (sat) p_sat = 1;
                if (s_rawout) fprintf(s_rawout, "# %s = %.2f GB/s over %lld ticks\n",
                                      name, t ? w / (double)t : 0.0, (long long)t);
            }
            return 0;
        });

        int64_t trd, twr;
        double total_gbps, duty[B_N];
        if (s_backend == BACKEND_AMC) {
            for (int b = 0; b < B_N; b++) duty[b] = 1.0;
            // total = MC aggregate (bare DCS), NOT the sum of per-requestor
            // buckets. Fall back to the bucket sum only if this machine exposes
            // no aggregate channel (then there is no duplicate to double-count).
            trd = s_agg_rd; twr = s_agg_wr;
            if (trd == 0 && twr == 0) {
                trd = s_rd[0] + s_rd[1] + s_rd[2] + s_rd[3];
                twr = s_wr[0] + s_wr[1] + s_wr[2] + s_wr[3];
            }
            total_gbps = dt > 0 ? (double)(trd + twr) / dt / 1e9 : 0.0;
        } else {
            // Normalise every bucket against the never-gated aggregate's tick
            // count, so a duty-cycled engine is time-averaged over the whole
            // interval rather than over the ticks it happened to be awake for.
            double base = (double)(p_agg_ticks > 0 ? p_agg_ticks : 0);
            for (int b = 0; b < B_N; b++) {
                double denom = base > 0 ? base : (double)p_ticks[b];
                double gbps_rd = denom > 0 ? p_wsum[b][0] / denom : 0.0;
                double gbps_wr = denom > 0 ? p_wsum[b][1] / denom : 0.0;
                s_rd[b] = (int64_t)(gbps_rd * 1e9 * dt);
                s_wr[b] = (int64_t)(gbps_wr * 1e9 * dt);
                duty[b] = base > 0 ? (double)p_ticks[b] / base : 1.0;
                if (duty[b] > 1.0) duty[b] = 1.0;
            }
            trd = s_rd[0] + s_rd[1] + s_rd[2] + s_rd[3];
            twr = s_wr[0] + s_wr[1] + s_wr[2] + s_wr[3];
            // The aggregate channel sees requestors the buckets do not
            // enumerate, so it is the better total; fall back to the buckets.
            total_gbps = p_agg_ticks > 0 ? p_agg_wsum / (double)p_agg_ticks
                                         : (dt > 0 ? (double)(trd + twr) / dt / 1e9 : 0.0);
        }

        fprintf(out, "%.6f,%.4f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.3f,"
                     "%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                wall, dt,
                (long long)s_rd[B_CPU], (long long)s_wr[B_CPU],
                (long long)s_rd[B_GPU], (long long)s_wr[B_GPU],
                (long long)s_rd[B_ANE], (long long)s_wr[B_ANE],
                (long long)s_rd[B_OTHER], (long long)s_wr[B_OTHER],
                (long long)trd, (long long)twr, total_gbps,
                duty[B_CPU], duty[B_GPU], duty[B_ANE], duty[B_OTHER],
                s_backend, p_sat);
        fflush(out);

        CFRelease(delta);
        CFRelease(prev);
        prev = cur;
        t_prev = t_cur;
    }
    if (out != stdout) fclose(out);
    return 0;
}

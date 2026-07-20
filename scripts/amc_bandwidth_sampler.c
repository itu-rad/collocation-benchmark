// ============================================================================
// AMC DRAM-bandwidth sampler (Apple Silicon, no root required)
//
// Streams per-engine DRAM traffic as CSV, one line per sampling interval, by
// reading the Apple Memory Controller's per-requestor byte counters from the
// private IOReport framework ("AMC Stats / Perf Counters" group; verified on
// M2 Pro / macOS 26 — see CONTENTION_EXPERIMENTS_REDESIGN.md §E3' Counters).
//
// Only the "DCS RD"/"DCS WR" channels are aggregated (memory-controller-side
// truth); summing every AMC agent would double-count fabric hops (ATC/SB/AFI).
// Requestors are bucketed: PCPU*/ECPU* -> cpu, GFX* -> gpu, ANE* -> ane,
// everything else -> other.
//
// CSV columns:
//   timestamp    wall-clock seconds (matches the framework CSV's %(created)f,
//                used for cross-process trace alignment), taken at interval end
//   dt_s         measured interval length (monotonic clock)
//   {cpu,gpu,ane,other}_{rd,wr}  bytes moved during the interval
//   total_rd, total_wr           sums of the above
//   total_gbps   (total_rd+total_wr) / dt_s / 1e9
//
// Usage:  amc_bandwidth_sampler [-i interval_ms] [-n samples] [-o out.csv] [--raw]
//         -n 0 (default) = run until SIGINT/SIGTERM (lines are flushed as
//         written, so the file is valid whenever the process is killed).
//         --raw additionally prints every non-zero AMC DCS channel per interval
//         (calibration/debugging).
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

static FnIterate Iterate;
static FnGetStr GetGroup, GetChannelName;
static FnGetInt GetInt;

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

// Accumulators live at file scope: blocks capture statics by plain reference
// (no __block byref cell — clang's byref support for aggregate types is what
// broke the first version of this loop).
static int64_t s_rd[B_N], s_wr[B_N];
static int64_t s_agg_rd, s_agg_wr;  // bare "DCS RD/WR" (no DIE) = MC aggregate = true total
static FILE* s_rawout;

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
    GetChannelName = (FnGetStr)dlsym(h, "IOReportChannelGetChannelName");
    GetInt = (FnGetInt)dlsym(h, "IOReportSimpleGetIntegerValue");
    if (!CreateSub || !CreateSamples || !SamplesDelta || !Iterate || !GetGroup || !GetChannelName || !GetInt) {
        fprintf(stderr, "dlsym failed (IOReport ABI changed?)\n"); return 1;
    }

    // Subscribe to the AMC group only (falls back to all channels if the
    // group-scoped copy is unavailable on this OS build).
    CFMutableDictionaryRef chans = NULL;
    if (CopyGroup) chans = CopyGroup(CFSTR("AMC Stats"), NULL, 0, 0, 0);
    if (!chans && CopyAll) chans = CopyAll(0, 0);
    if (!chans) { fprintf(stderr, "IOReport channel copy failed\n"); return 1; }

    CFMutableDictionaryRef sub_chans = NULL;
    void* sub = CreateSub(NULL, chans, &sub_chans, 0, NULL);
    if (!sub) { fprintf(stderr, "IOReportCreateSubscription failed\n"); return 1; }

    // Verify AMC DCS channels actually exist before promising data.
    __block int n_dcs = 0;
    Iterate((CFDictionaryRef)chans, ^int(CFDictionaryRef ch) {
        char g[128], name[256];
        cstr(GetGroup(ch), g, sizeof g); cstr(GetChannelName(ch), name, sizeof name);
        if (strstr(g, "AMC") && (strstr(name, "DCS RD") || strstr(name, "DCS WR"))) n_dcs++;
        return 0;
    });
    if (n_dcs == 0) {
        fprintf(stderr, "no 'AMC Stats' DCS RD/WR channels on this machine -- "
                        "run scripts/preflight_bandwidth_counters.sh; falling back is not supported\n");
        return 3;
    }

    FILE* out = stdout;
    if (out_path) {
        out = fopen(out_path, "w");
        if (!out) { perror("fopen"); return 1; }
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    fprintf(out, "timestamp,dt_s,cpu_rd,cpu_wr,gpu_rd,gpu_wr,ane_rd,ane_wr,"
                 "other_rd,other_wr,total_rd,total_wr,total_gbps\n");
    fflush(out);
    fprintf(stderr, "amc_bandwidth_sampler: %d DCS channels, interval %ld ms, %s\n",
            n_dcs, interval_ms, out_path ? out_path : "stdout");

    CFDictionaryRef prev = CreateSamples(sub, sub_chans, NULL);
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
        s_agg_rd = s_agg_wr = 0;
        s_rawout = raw ? out : NULL;
        Iterate(delta, ^int(CFDictionaryRef ch) {
            char g[128], name[256];
            cstr(GetGroup(ch), g, sizeof g);
            if (!strstr(g, "AMC")) return 0;
            cstr(GetChannelName(ch), name, sizeof name);
            int is_rd = strstr(name, "DCS RD") != NULL;
            int is_wr = strstr(name, "DCS WR") != NULL;
            if (!is_rd && !is_wr) return 0;
            int64_t v = GetInt(ch, NULL);
            if (v < 0) return 0;  // counter wrap / bogus sample
            // The bare "DCS RD/WR" channel (no DIE/requestor prefix) is the
            // memory-controller AGGREGATE = the true total. Per-requestor
            // channels ("DIE0 ECPU0 DCS RD", "DIE0 GFX DCS RD", ...) are the
            // attribution breakdown. Summing the aggregate AND its components
            // double-counts (the calibration bug: reported total = 2x real).
            if (!strstr(name, "DIE")) {
                if (is_rd) s_agg_rd += v; else s_agg_wr += v;
            } else {
                int b = B_OTHER;
                if (strstr(name, "PCPU") || strstr(name, "ECPU")) b = B_CPU;
                else if (strstr(name, "GFX")) b = B_GPU;
                else if (strstr(name, "ANE") || strstr(name, "ANS")) b = B_ANE;
                if (is_rd) s_rd[b] += v; else s_wr[b] += v;
            }
            if (s_rawout && v > 0) fprintf(s_rawout, "# %s = %lld\n", name, (long long)v);
            return 0;
        });

        // total = MC aggregate (bare DCS), NOT the sum of per-requestor buckets.
        // Fall back to the bucket sum only if this machine exposes no aggregate
        // channel (then the double-count risk is absent because there is no
        // duplicate).
        int64_t trd = s_agg_rd, twr = s_agg_wr;
        if (trd == 0 && twr == 0) {
            trd = s_rd[0] + s_rd[1] + s_rd[2] + s_rd[3];
            twr = s_wr[0] + s_wr[1] + s_wr[2] + s_wr[3];
        }
        fprintf(out, "%.6f,%.4f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.3f\n",
                wall, dt,
                (long long)s_rd[B_CPU], (long long)s_wr[B_CPU],
                (long long)s_rd[B_GPU], (long long)s_wr[B_GPU],
                (long long)s_rd[B_ANE], (long long)s_wr[B_ANE],
                (long long)s_rd[B_OTHER], (long long)s_wr[B_OTHER],
                (long long)trd, (long long)twr,
                dt > 0 ? (double)(trd + twr) / dt / 1e9 : 0.0);
        fflush(out);

        CFRelease(delta);
        CFRelease(prev);
        prev = cur;
        t_prev = t_cur;
    }
    if (out != stdout) fclose(out);
    return 0;
}

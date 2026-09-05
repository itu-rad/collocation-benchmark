// DRAM-bandwidth channel probe via IOReport (Apple Silicon, no root needed).
//
// Enumerates every IOReport channel and reports, per group, how many channels
// survive each stage: COPIED -> SUBSCRIBED -> SAMPLED. That census is the point:
// a group can enumerate fully and still contribute nothing, because the driver
// refuses the subscription and the library drops its channels silently. That is
// exactly what "AMC Stats" does on M3-family silicon, and reading values without
// checking survival made it look like counters that "exist and read zero".
//
// Prints 1s deltas for bandwidth-like channels, using the RIGHT accessor for
// each channel's format:
//   fmt 1 (simple)  -> IOReportSimpleGetIntegerValue        (M2 "AMC Stats" bytes)
//   fmt 2 (state)   -> IOReportStateGet*                    (M3 "PMP / DCS BW"
//                      32-bin bandwidth histograms; the simple accessor returns
//                      garbage for these, which is why they were missed)
//
// See docs/amc-m3-counters-plan.md. Part of the E3'/E6' counter pre-flight.
// Build: clang -o ioreport_bw_probe scripts/ioreport_bw_probe.c -framework CoreFoundation
// Usage: ioreport_bw_probe [--all]      (--all also lists non-bandwidth channels)
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>

typedef CFMutableDictionaryRef (*FnCopyAll)(uint64_t, uint64_t);
typedef void* (*FnCreateSub)(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*FnCreateSamples)(void*, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*FnSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef int (*FnIterate)(CFDictionaryRef, int (^)(CFDictionaryRef));
typedef CFStringRef (*FnGetStr)(CFDictionaryRef);
typedef int64_t (*FnGetInt)(CFDictionaryRef, void*);  // 2nd arg ABI is unstable across headers; only NULL is safe
typedef int32_t (*FnI32)(CFDictionaryRef);
typedef int64_t (*FnStateRes)(CFDictionaryRef, int32_t);
typedef CFStringRef (*FnStateName)(CFDictionaryRef, int32_t);

static FnIterate Iterate;
static FnGetStr GetGroup, GetSubGroup, GetChannelName, GetUnitLabel, GetDriverName;
static FnGetInt GetInt;
static FnI32 GetFormat, StateCount;
static FnStateRes StateRes;
static FnStateName StateName;

static void cstr(CFStringRef s, char* buf, size_t n) {
    buf[0] = 0;
    if (s) CFStringGetCString(s, buf, n, kCFStringEncodingUTF8);
}
static int is_bwish(const char* g, const char* sg, const char* ch, const char* u) {
    char all[1024];
    snprintf(all, sizeof all, "%s|%s|%s|%s", g, sg, ch, u);
    for (char* p = all; *p; p++) *p = (char)tolower((unsigned char)*p);
    return strstr(all, "bandwidth") || strstr(all, "dcs") || strstr(all, "amcc")
        || strstr(all, "bw") || strstr(all, "byte") || strstr(all, "mib")
        || strstr(all, "gib") || strstr(all, " rd") || strstr(all, "rd ")
        || strstr(all, " wr") || strstr(all, "wr ") != NULL;
}

// Mean bandwidth of a 32-bin "<N>GB/s" histogram. Returns 0 if the channel is
// not shaped like one. Bin i is labelled with its UPPER edge; residency is ticks.
static int hist_mean(CFDictionaryRef ch, double* mean, long long* ticks, int* sat) {
    *mean = 0; *ticks = 0; *sat = 0;
    if (!StateCount) return 0;
    int32_t c = StateCount(ch);
    if (c <= 0 || c > 64) return 0;
    double wsum = 0; long long tot = 0; int looks_like_bw = 0;
    for (int32_t i = 0; i < c; i++) {
        char sn[128]; cstr(StateName(ch, i), sn, sizeof sn);
        if (strstr(sn, "GB/s")) looks_like_bw = 1;
        double label = atof(sn);
        int64_t r = StateRes(ch, i);
        if (r <= 0 || label <= 0) continue;
        double width = label / (double)(i + 1);
        wsum += (double)r * (label - width / 2.0);
        tot += r;
        if (i == c - 1) *sat = 1;
    }
    if (!looks_like_bw) return 0;
    *ticks = tot;
    *mean = tot ? wsum / (double)tot : 0.0;
    return 1;
}

// ---- per-group census -------------------------------------------------------
// Keyed "group  <<driver>>" so two instances of a driver stay distinct.
static CFMutableDictionaryRef tally(CFDictionaryRef d) {
    CFMutableDictionaryRef t = CFDictionaryCreateMutable(NULL, 0,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (!d) return t;
    Iterate(d, ^int(CFDictionaryRef ch) {
        char gb[256], db[256];
        cstr(GetGroup(ch), gb, sizeof gb);
        cstr(GetDriverName ? GetDriverName(ch) : NULL, db, sizeof db);
        char key[600]; snprintf(key, sizeof key, "%s  <<%s>>", gb, db);
        CFStringRef k = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
        if (!k) return 0;
        CFNumberRef prev = CFDictionaryGetValue(t, k);
        int n = 0; if (prev) CFNumberGetValue(prev, kCFNumberIntType, &n);
        n++;
        CFNumberRef nv = CFNumberCreate(NULL, kCFNumberIntType, &n);
        CFDictionarySetValue(t, k, nv);
        CFRelease(nv); CFRelease(k);
        return 0;
    });
    return t;
}
static int count_of(CFDictionaryRef t, CFStringRef k) {
    CFNumberRef v = CFDictionaryGetValue(t, k);
    int n = 0; if (v) CFNumberGetValue(v, kCFNumberIntType, &n);
    return n;
}

int main(int argc, char** argv) {
    void* h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) h = dlopen("IOReport.framework/IOReport", RTLD_NOW);
    if (!h) { printf("dlopen libIOReport FAILED: %s\n", dlerror()); return 1; }
    FnCopyAll CopyAll = (FnCopyAll)dlsym(h, "IOReportCopyAllChannels");
    FnCreateSub CreateSub = (FnCreateSub)dlsym(h, "IOReportCreateSubscription");
    FnCreateSamples CreateSamples = (FnCreateSamples)dlsym(h, "IOReportCreateSamples");
    FnSamplesDelta SamplesDelta = (FnSamplesDelta)dlsym(h, "IOReportCreateSamplesDelta");
    Iterate = (FnIterate)dlsym(h, "IOReportIterate");
    GetGroup = (FnGetStr)dlsym(h, "IOReportChannelGetGroup");
    GetSubGroup = (FnGetStr)dlsym(h, "IOReportChannelGetSubGroup");
    GetChannelName = (FnGetStr)dlsym(h, "IOReportChannelGetChannelName");
    GetUnitLabel = (FnGetStr)dlsym(h, "IOReportChannelGetUnitLabel");
    GetDriverName = (FnGetStr)dlsym(h, "IOReportChannelGetDriverName");
    GetInt = (FnGetInt)dlsym(h, "IOReportSimpleGetIntegerValue");
    GetFormat = (FnI32)dlsym(h, "IOReportChannelGetFormat");
    StateCount = (FnI32)dlsym(h, "IOReportStateGetCount");
    StateRes = (FnStateRes)dlsym(h, "IOReportStateGetResidency");
    StateName = (FnStateName)dlsym(h, "IOReportStateGetNameForIndex");
    if (!CopyAll || !Iterate || !GetGroup) { printf("dlsym failed\n"); return 1; }
    printf("libIOReport loaded OK\n");

    CFMutableDictionaryRef chans = CopyAll(0, 0);
    if (!chans) { printf("IOReportCopyAllChannels returned NULL\n"); return 1; }

    __block int total = 0, flagged = 0;
    int list_all = (argc > 1 && !strcmp(argv[1], "--all"));
    Iterate((CFDictionaryRef)chans, ^int(CFDictionaryRef ch) {
        char g[256], sg[256], name[256], u[128];
        cstr(GetGroup(ch), g, sizeof g); cstr(GetSubGroup(ch), sg, sizeof sg);
        cstr(GetChannelName(ch), name, sizeof name); cstr(GetUnitLabel(ch), u, sizeof u);
        total++;
        int f = is_bwish(g, sg, name, u);
        if (f) { flagged++; printf("FLAG [%s / %s / %s] unit=%s fmt=%d\n", g, sg, name, u,
                                   GetFormat ? GetFormat(ch) : -1); }
        else if (list_all) printf("     [%s / %s / %s] unit=%s\n", g, sg, name, u);
        return 0;
    });
    printf("--- %d channels total, %d bandwidth-like ---\n", total, flagged);

    CFMutableDictionaryRef sub_chans = NULL;
    void* sub = CreateSub ? CreateSub(NULL, chans, &sub_chans, 0, NULL) : NULL;
    printf("\n--- subscription: %s ---\n", sub ? "ok" : "FAILED");
    CFDictionaryRef s1 = NULL, s2 = NULL, d = NULL;
    if (sub) {
        s1 = CreateSamples(sub, sub_chans, NULL);
        sleep(1);
        s2 = CreateSamples(sub, sub_chans, NULL);
        d = SamplesDelta(s1, s2, NULL);
    }

    // The decisive census. A group that is copied and "subscribed" but never
    // sampled contributes nothing, no matter what its channels are named.
    printf("\n--- per-group survival: COPIED -> SUBSCRIBED -> SAMPLED ---\n");
    CFMutableDictionaryRef ta = tally((CFDictionaryRef)chans);
    CFMutableDictionaryRef tb = tally((CFDictionaryRef)sub_chans);
    CFMutableDictionaryRef tc = tally(s1);
    CFIndex n = CFDictionaryGetCount(ta);
    const void** keys = malloc((size_t)n * sizeof(void*));
    CFDictionaryGetKeysAndValues(ta, keys, NULL);
    int dropped = 0;
    printf("%-7s %-7s %-7s  %s\n", "COPIED", "SUBBED", "SAMPLED", "group  <<driver>>");
    for (CFIndex i = 0; i < n; i++) {
        CFStringRef k = keys[i];
        int a = count_of(ta, k), b = count_of(tb, k), c = count_of(tc, k);
        char kb[600]; cstr(k, kb, sizeof kb);
        int drop = (a > 0 && c == 0);
        if (drop) dropped++;
        if (drop || list_all || strstr(kb, "AMC") || strstr(kb, "PMP"))
            printf("%-7d %-7d %-7d  %s%s\n", a, b, c, kb, drop ? "   <== DROPPED" : "");
    }
    free(keys);
    if (dropped)
        printf("  (%d group(s) enumerate but never reach a sample -- the driver refused\n"
               "   the subscription. Their channels are UNREADABLE, not zero.)\n", dropped);

    if (d) {
        printf("\n--- 1s deltas for flagged channels (format-aware) ---\n");
        Iterate(d, ^int(CFDictionaryRef ch) {
            char g[256], sg[256], name[256], u[128];
            cstr(GetGroup(ch), g, sizeof g); cstr(GetSubGroup(ch), sg, sizeof sg);
            cstr(GetChannelName(ch), name, sizeof name); cstr(GetUnitLabel(ch), u, sizeof u);
            if (!is_bwish(g, sg, name, u)) return 0;
            double mean; long long ticks; int sat;
            if (hist_mean(ch, &mean, &ticks, &sat)) {
                if (ticks == 0) return 0;   // gated block: no ticks, nothing to say
                printf("  [%s / %s / %s] %.2f GB/s over %lld ticks%s\n",
                       g, sg, name, mean, ticks, sat ? "  SATURATED (top bin)" : "");
            } else {
                int64_t v = GetInt ? GetInt(ch, NULL) : 0;
                if (v == INT64_MIN) return 0;   // not a simple channel
                printf("  [%s / %s / %s] delta=%lld unit=%s\n", g, sg, name, (long long)v, u);
            }
            return 0;
        });
    }
    return 0;
}

// AMC (Apple Memory Controller) DRAM-bandwidth probe via IOReport (no root needed).
// Enumerates per-requestor RD/WR byte counters (CPU/GPU/ANE/...), prints 1s deltas.
// Part of the E3'/E6' counter pre-flight — see CONTENTION_EXPERIMENTS_REDESIGN.md.
// Build: clang -o ioreport_bw_probe scripts/ioreport_bw_probe.c -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <stdio.h>
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

static FnGetStr GetGroup, GetSubGroup, GetChannelName, GetUnitLabel;

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

int main(int argc, char** argv) {
    void* h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) h = dlopen("IOReport.framework/IOReport", RTLD_NOW);
    if (!h) { printf("dlopen libIOReport FAILED: %s\n", dlerror()); return 1; }
    FnCopyAll CopyAll = (FnCopyAll)dlsym(h, "IOReportCopyAllChannels");
    FnCreateSub CreateSub = (FnCreateSub)dlsym(h, "IOReportCreateSubscription");
    FnCreateSamples CreateSamples = (FnCreateSamples)dlsym(h, "IOReportCreateSamples");
    FnSamplesDelta SamplesDelta = (FnSamplesDelta)dlsym(h, "IOReportCreateSamplesDelta");
    FnIterate Iterate = (FnIterate)dlsym(h, "IOReportIterate");
    GetGroup = (FnGetStr)dlsym(h, "IOReportChannelGetGroup");
    GetSubGroup = (FnGetStr)dlsym(h, "IOReportChannelGetSubGroup");
    GetChannelName = (FnGetStr)dlsym(h, "IOReportChannelGetChannelName");
    GetUnitLabel = (FnGetStr)dlsym(h, "IOReportChannelGetUnitLabel");
    FnGetInt GetInt = (FnGetInt)dlsym(h, "IOReportSimpleGetIntegerValue");
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
        if (f) { flagged++; printf("FLAG [%s / %s / %s] unit=%s\n", g, sg, name, u); }
        else if (list_all) printf("     [%s / %s / %s] unit=%s\n", g, sg, name, u);
        return 0;
    });
    printf("--- %d channels total, %d bandwidth-like ---\n", total, flagged);

    printf("--- groups ---\n");
    CFMutableSetRef seen = CFSetCreateMutable(NULL, 0, &kCFTypeSetCallBacks);
    Iterate((CFDictionaryRef)chans, ^int(CFDictionaryRef ch) {
        CFStringRef gs = GetGroup(ch);
        if (gs && !CFSetContainsValue(seen, gs)) {
            CFSetAddValue(seen, gs);
            char g[256]; cstr(gs, g, sizeof g);
            printf("  %s\n", g);
        }
        return 0;
    });

    if (flagged && CreateSub && CreateSamples && SamplesDelta && GetInt) {
        CFMutableDictionaryRef sub_chans = NULL;
        void* sub = CreateSub(NULL, chans, &sub_chans, 0, NULL);
        if (sub) {
            CFDictionaryRef s1 = CreateSamples(sub, sub_chans, NULL);
            sleep(1);
            CFDictionaryRef s2 = CreateSamples(sub, sub_chans, NULL);
            CFDictionaryRef d = SamplesDelta(s1, s2, NULL);
            printf("--- 1s deltas for flagged channels ---\n");
            Iterate(d, ^int(CFDictionaryRef ch) {
                char g[256], sg[256], name[256], u[128];
                cstr(GetGroup(ch), g, sizeof g); cstr(GetSubGroup(ch), sg, sizeof sg);
                cstr(GetChannelName(ch), name, sizeof name); cstr(GetUnitLabel(ch), u, sizeof u);
                if (!is_bwish(g, sg, name, u)) return 0;
                int64_t v = GetInt(ch, NULL);
                printf("  [%s / %s / %s] delta=%lld unit=%s\n", g, sg, name, (long long)v, u);
                return 0;
            });
        }
    }
    return 0;
}

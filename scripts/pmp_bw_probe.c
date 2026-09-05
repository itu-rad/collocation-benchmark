// PMP "DCS BW" histogram probe: per-requestor DRAM bandwidth (Apple silicon, no root).
//
// The M3-family replacement for the "AMC Stats" byte counters, which cannot be
// subscribed on M3 (see scripts/ior_stage.c and docs/amc-m3-counters-plan.md).
// Channels live in group PMP, subgroup "DCS BW", one per requestor:
// EACC0 / PACC0 (cpu), AGX (gpu), ANE0 (ane), ISP/DISP/... (other), plus a bare
// "RD+WR" controller aggregate.
//
// These are kIOReportFormatState (fmt=2) channels used as 32-bin histograms:
// bin i is labelled with its UPPER EDGE in GB/s and its residency is the number
// of PMP ticks spent in that bandwidth bin during the interval. Mean bandwidth
// is sum(residency_i * midpoint_i) / sum(residency_i). Per-requestor bins are
// 1 GB/s wide (saturates above 32 GB/s); the aggregate is 2 GB/s (above 64).
// IOReportSimpleGetIntegerValue returns garbage for these -- use the state
// accessors, as below.
//
// Usage: pmp_bw_probe [n_intervals] [interval_s] [name_filter]
// Build: clang -O2 -o pmp_bw_probe scripts/pmp_bw_probe.c -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
typedef CFMutableDictionaryRef (*FnCopyAll)(uint64_t,uint64_t);
typedef void* (*FnCreateSub)(void*,CFMutableDictionaryRef,CFMutableDictionaryRef*,uint64_t,CFTypeRef);
typedef CFDictionaryRef (*FnCreateSamples)(void*,CFMutableDictionaryRef,CFTypeRef);
typedef CFDictionaryRef (*FnSamplesDelta)(CFDictionaryRef,CFDictionaryRef,CFTypeRef);
typedef int (*FnIterate)(CFDictionaryRef,int(^)(CFDictionaryRef));
typedef CFStringRef (*FnGetStr)(CFDictionaryRef);
typedef int32_t (*FnI32)(CFDictionaryRef);
typedef int64_t (*FnStateRes)(CFDictionaryRef,int32_t);
typedef CFStringRef (*FnStateName)(CFDictionaryRef,int32_t);
static FnIterate Iterate; static FnGetStr GetGroup,GetSubGroup,GetName;
static FnI32 StateCount; static FnStateRes StateRes; static FnStateName StateName;
static void cstr(CFStringRef s,char*b,size_t n){b[0]=0; if(s)CFStringGetCString(s,b,n,kCFStringEncodingUTF8);}
static double now_mono(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static const char* g_filter="DCS BW";
int main(int argc,char**argv){
    int n=(argc>1)?atoi(argv[1]):3; double iv=(argc>2)?atof(argv[2]):1.0;
    if(argc>3) g_filter=argv[3];
    void* h=dlopen("/usr/lib/libIOReport.dylib",RTLD_NOW);
    FnCopyAll CopyAll=(FnCopyAll)dlsym(h,"IOReportCopyAllChannels");
    FnCreateSub CreateSub=(FnCreateSub)dlsym(h,"IOReportCreateSubscription");
    FnCreateSamples CreateSamples=(FnCreateSamples)dlsym(h,"IOReportCreateSamples");
    FnSamplesDelta SamplesDelta=(FnSamplesDelta)dlsym(h,"IOReportCreateSamplesDelta");
    Iterate=(FnIterate)dlsym(h,"IOReportIterate");
    GetGroup=(FnGetStr)dlsym(h,"IOReportChannelGetGroup");
    GetSubGroup=(FnGetStr)dlsym(h,"IOReportChannelGetSubGroup");
    GetName=(FnGetStr)dlsym(h,"IOReportChannelGetChannelName");
    StateCount=(FnI32)dlsym(h,"IOReportStateGetCount");
    StateRes=(FnStateRes)dlsym(h,"IOReportStateGetResidency");
    StateName=(FnStateName)dlsym(h,"IOReportStateGetNameForIndex");
    CFMutableDictionaryRef all=CopyAll(0,0); CFMutableDictionaryRef sc=NULL;
    void* sub=CreateSub(NULL,all,&sc,0,NULL); if(!sub){printf("sub failed\n");return 1;}
    CFDictionaryRef prev=CreateSamples(sub,sc,NULL); double tp=now_mono();
    printf("%-22s %8s %8s %10s %12s\n","channel","ticks","act_tk","meanGBps","sumGB/s*tk");
    for(int k=0;k<n;k++){
        struct timespec ts={(time_t)iv,(long)((iv-(long)iv)*1e9)}; nanosleep(&ts,NULL);
        CFDictionaryRef cur=CreateSamples(sub,sc,NULL); double tc=now_mono();
        CFDictionaryRef d=SamplesDelta(prev,cur,NULL); double dt=tc-tp;
        printf("--- interval %d  dt=%.3fs ---\n",k,dt);
        Iterate(d,^int(CFDictionaryRef ch){
            char g[128],sg[128],nm[128];
            cstr(GetGroup(ch),g,sizeof g); cstr(GetSubGroup(ch),sg,sizeof sg); cstr(GetName(ch),nm,sizeof nm);
            char key[400]; snprintf(key,sizeof key,"%s / %s / %s",g,sg,nm);
            if(!strstr(key,g_filter)) return 0;
            int32_t c=StateCount?StateCount(ch):0; if(c<=0||c>64) return 0;
            double wsum=0; long long tot=0, act=0;
            for(int32_t i=0;i<c;i++){
                char sn[128]; cstr(StateName(ch,i),sn,sizeof sn);
                double lbl=atof(sn); long long r=StateRes(ch,i);
                if(r<0) r=0;
                tot+=r; if(i>0) act+=r;
                double width = lbl/(double)(i+1);   // uniform bins: label=(i+1)*w
                wsum += r * (lbl - width/2.0);
            }
            if(tot==0) return 0;
            printf("%-22s %8lld %8lld %10.2f %12.0f\n", nm, tot, act, wsum/(double)tot, wsum);
            return 0;});
        CFRelease(d); CFRelease(prev); prev=cur; tp=tc;
    }
    return 0;
}

// IOReport subscription-stage census: counts AMC Stats channels at each stage
// (copied -> subscribed -> sampled) via three subscription paths.
//
// Diagnoses the M3 Pro per-engine DRAM counter failure (docs/amc-m3-counters-plan.md):
// on M2 Pro all 117 AMC channels survive into the sample; on M3 Pro all 128 are
// copied and "subscribed" but ZERO reach the sample -- AppleH15MemCacheController
// refuses the subscription, and IOReportCreateSubscription drops them silently
// when the request is built from the all-channels dict.
//
// Build: clang -O2 -o ior_stage scripts/ior_stage.c -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
typedef CFMutableDictionaryRef (*FnCopyAll)(uint64_t, uint64_t);
typedef CFMutableDictionaryRef (*FnCopyGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void* (*FnCreateSub)(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*FnCreateSamples)(void*, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*FnSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef int (*FnIterate)(CFDictionaryRef, int (^)(CFDictionaryRef));
typedef CFStringRef (*FnGetStr)(CFDictionaryRef);
typedef int64_t (*FnGetInt)(CFDictionaryRef, void*);
static FnIterate Iterate; static FnGetStr GetGroup,GetSub,GetName; static FnGetInt GetI;
static void cstr(CFStringRef s,char*b,size_t n){b[0]=0; if(s)CFStringGetCString(s,b,n,kCFStringEncodingUTF8);}
static int count(CFDictionaryRef d, const char* tag, int show){
    __block int n=0, nz=0;
    if (!d) { printf("%-28s NULL\n", tag); return 0; }
    Iterate(d, ^int(CFDictionaryRef ch){
        char g[256],nm[256]; cstr(GetGroup(ch),g,sizeof g); cstr(GetName(ch),nm,sizeof nm);
        if (strstr(g,"AMC Stats")) { n++;
            int64_t v = GetI?GetI(ch,NULL):0; if (v) nz++;
            if (show && n<=6) printf("      e.g. [%s] %s = %lld\n", g, nm, (long long)v); }
        return 0; });
    printf("%-28s AMC Stats channels = %d (nonzero=%d)\n", tag, n, nz);
    return n;
}
int main(void){
    void* h=dlopen("/usr/lib/libIOReport.dylib",RTLD_NOW);
    FnCopyAll CopyAll=(FnCopyAll)dlsym(h,"IOReportCopyAllChannels");
    FnCopyGroup CopyGroup=(FnCopyGroup)dlsym(h,"IOReportCopyChannelsInGroup");
    FnCreateSub CreateSub=(FnCreateSub)dlsym(h,"IOReportCreateSubscription");
    FnCreateSamples CreateSamples=(FnCreateSamples)dlsym(h,"IOReportCreateSamples");
    FnSamplesDelta SamplesDelta=(FnSamplesDelta)dlsym(h,"IOReportCreateSamplesDelta");
    Iterate=(FnIterate)dlsym(h,"IOReportIterate");
    GetGroup=(FnGetStr)dlsym(h,"IOReportChannelGetGroup");
    GetSub=(FnGetStr)dlsym(h,"IOReportChannelGetSubGroup");
    GetName=(FnGetStr)dlsym(h,"IOReportChannelGetChannelName");
    GetI=(FnGetInt)dlsym(h,"IOReportSimpleGetIntegerValue");

    printf("### PATH A: CopyAllChannels -> subscribe -> sample\n");
    CFMutableDictionaryRef all=CopyAll(0,0);
    count((CFDictionaryRef)all,"CopyAllChannels",1);
    CFMutableDictionaryRef subch=NULL;
    void* sub=CreateSub(NULL,all,&subch,0,NULL);
    printf("  CreateSubscription: %s\n", sub?"ok":"FAILED");
    count((CFDictionaryRef)subch,"  subbedChannels",0);
    if(sub){
        CFDictionaryRef s1=CreateSamples(sub,subch,NULL);
        count(s1,"  sample1",1);
        sleep(1);
        CFDictionaryRef s2=CreateSamples(sub,subch,NULL);
        CFDictionaryRef d=SamplesDelta(s1,s2,NULL);
        count(d,"  delta",1);
    }
    printf("\n### PATH B: CopyChannelsInGroup(\"AMC Stats\") -> subscribe -> sample\n");
    CFMutableDictionaryRef grp=CopyGroup?CopyGroup(CFSTR("AMC Stats"),NULL,0,0,0):NULL;
    count((CFDictionaryRef)grp,"CopyChannelsInGroup",1);
    if(grp){
        CFMutableDictionaryRef subch2=NULL;
        void* sub2=CreateSub(NULL,grp,&subch2,0,NULL);
        printf("  CreateSubscription: %s\n", sub2?"ok":"FAILED");
        count((CFDictionaryRef)subch2,"  subbedChannels",0);
        if(sub2){
            CFDictionaryRef s1=CreateSamples(sub2,subch2,NULL);
            count(s1,"  sample1",1);
            sleep(1);
            CFDictionaryRef s2=CreateSamples(sub2,subch2,NULL);
            CFDictionaryRef d=SamplesDelta(s1,s2,NULL);
            count(d,"  delta",1);
        }
    }
    printf("\n### PATH C: CopyChannelsInGroup with subgroup \"Perf Counters\"\n");
    CFMutableDictionaryRef grp2=CopyGroup?CopyGroup(CFSTR("AMC Stats"),CFSTR("Perf Counters"),0,0,0):NULL;
    count((CFDictionaryRef)grp2,"CopyChannelsInGroup+sg",1);
    if(grp2){
        CFMutableDictionaryRef subch3=NULL;
        void* sub3=CreateSub(NULL,grp2,&subch3,0,NULL);
        printf("  CreateSubscription: %s\n", sub3?"ok":"FAILED");
        if(sub3){
            CFDictionaryRef s1=CreateSamples(sub3,subch3,NULL);
            sleep(1);
            CFDictionaryRef s2=CreateSamples(sub3,subch3,NULL);
            count(SamplesDelta(s1,s2,NULL),"  delta",1);
        }
    }
    return 0;
}

import pandas as pd
import glob
import os
import re

def analyze_file(filepath):
    try:
        latencies = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Store start times for each query
        # Key: (query_id, epoch, batch)
        query_starts = {}
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
                
            try:
                ts = float(parts[0])
            except ValueError:
                continue

            component = parts[2]
            event = parts[4]
            
            if component.startswith("pipeline -"):
                if len(parts) < 9:
                    continue
                query_id = parts[5]
                epoch = parts[7]
                batch = parts[8]
                key = (query_id, epoch, batch)
                
                if event == "start":
                    query_starts[key] = ts
                elif event == "end":
                    if key in query_starts:
                        latencies.append(ts - query_starts[key])
                        del query_starts[key]
        
        if not latencies:
            return None
            
        avg_latency = sum(latencies) / len(latencies)
        return avg_latency
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def generate_latex_depth():
    results = {}
    
    # We want to match noop_chain_depth_X_size_0_mode_ref.csv
    # and also fallback to noop_chain_depth_X.csv if the specific one doesn't exist
    files = glob.glob("collocation-benchmark/evaluation/results/noop_chain_depth_*.csv")
    
    # Organize by depth
    depth_data = {}
    for f in files:
        # Try specific pattern first
        match_spec = re.search(r"depth_(\d+)_size_0_mode_ref\.csv", f)
        if match_spec:
            depth = int(match_spec.group(1))
            depth_data[depth] = f
            continue
            
        # Fallback pattern (only if not already found)
        match_base = re.search(r"depth_(\d+)\.csv", f)
        if match_base:
            depth = int(match_base.group(1))
            if depth not in depth_data:
                depth_data[depth] = f

    for depth, f in depth_data.items():
        avg_latency = analyze_file(f)
        if avg_latency is not None:
            results[depth] = avg_latency

    print("\n% --- Table: Depth Scaling Results ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print("Depth & Avg. Latency (ms) & Latency per Stage (\\mu s) \\\\")
    print("\\hline")
    for depth in sorted(results.keys()):
        latency_ms = results[depth] * 1000
        latency_per_stage_us = (results[depth] / depth) * 1e6
        print(f"{depth} & {latency_ms:.3f} & {latency_per_stage_us:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Choreo framework overhead across different pipeline depths.}")
    print("\\label{tab:depth_scaling}")
    print("\\end{table}")

def generate_latex_payload():
    results = {}
    files = glob.glob("collocation-benchmark/evaluation/results/noop_chain_depth_10_size_*.csv")
    
    for f in files:
        match = re.search(r"size_(\d+)_mode_(\w+)\.csv", f)
        if match:
            size = int(match.group(1))
            mode = match.group(2)
            if size == 0: continue # Skip size 0 as it's the baseline for depth
            avg_latency = analyze_file(f)
            if avg_latency is not None:
                if size not in results:
                    results[size] = {}
                results[size][mode] = avg_latency

    print("\n% --- Table: Zero-Copy Performance ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("Payload Size & Ref Latency (ms) & Copy Latency (ms) & Speedup \\\\")
    print("\\hline")
    
    size_labels = {
        1024: "1 KB",
        1048576: "1 MB",
        10485760: "10 MB"
    }
    
    for size in sorted(results.keys()):
        ref = results[size].get("ref")
        copy = results[size].get("copy")
        if ref and copy:
            speedup = copy / ref
            label = size_labels.get(size, str(size))
            print(f"{label} & {ref*1000:.3f} & {copy*1000:.3f} & {speedup:.2f}x \\\\")
            
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of Choreo's zero-copy reference passing vs. deep copying across payload sizes (Depth=10).}")
    print("\\label{tab:zero_copy}")
    print("\\end{table}")

if __name__ == "__main__":
    generate_latex_depth()
    generate_latex_payload()

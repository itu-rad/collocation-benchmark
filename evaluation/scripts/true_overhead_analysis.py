import os
import numpy as np

def get_choreo_latencies(filepath):
    starts = []
    ends = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5: continue
            if parts[2].startswith("pipeline -"):
                ts = float(parts[0])
                event = parts[4]
                if event == "start": starts.append(ts)
                elif event == "end": ends.append(ts)
    
    min_len = min(len(starts), len(ends))
    return np.array(ends[:min_len]) - np.array(starts[:min_len])

def get_baseline_latencies(filepath):
    ends = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5: continue
            if parts[2] == "training_step":
                ts = float(parts[0])
                event = parts[4]
                if event == "end": ends.append(ts)
    
    if len(ends) < 2: return np.array([])
    return np.diff(ends)

choreo_file = "collocation-benchmark/evaluation/results/efficientnetv2_s_imagenette_training.csv"
baseline_file = "collocation-benchmark/evaluation/results/baseline_finetune.csv"

c_lat = get_choreo_latencies(choreo_file) * 1000
b_lat = get_baseline_latencies(baseline_file) * 1000

# Discard first 100 warmup
if len(c_lat) > 100: c_lat = c_lat[100:]
if len(b_lat) > 100: b_lat = b_lat[100:]

if len(c_lat) > 0 and len(b_lat) > 0:
    print("\n--- TRUE End-to-End Comparison (Apples-to-Apples) ---")
    print("Task: Process 1 Batch (Load + Compute + Sync)")
    print(f"{ 'Metric':<25} | { 'Monolithic':<12} | { 'Choreo':<12} | { 'Overhead %':<10}")
    print("-" * 75)
    
    metrics = [
        ("Mean Latency (ms)", np.mean(b_lat), np.mean(c_lat)),
        ("Median Latency (ms)", np.median(b_lat), np.median(c_lat)),
        ("P99 Latency (ms)", np.percentile(b_lat, 99), np.percentile(c_lat, 99)),
        ("Std Dev (ms)", np.std(b_lat), np.std(c_lat))
    ]
    
    for name, b_val, c_val in metrics:
        ovh = ((c_val - b_val) / b_val) * 100
        print(f"{ name:<25} | { b_val:<12.2f} | { c_val:<12.2f} | { ovh:<9.2f}%")
else:
    print(f"Insufficient data. Choreo: {len(c_lat)}, Baseline: {len(b_lat)}")

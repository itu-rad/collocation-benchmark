import pandas as pd
import os
import numpy as np

def analyze_file(filepath, step_name):
    try:
        df = pd.read_csv(filepath, header=None, names=["timestamp", "parent", "module", "phase", "event"], usecols=[0, 1, 2, 3, 4], on_bad_lines='skip')
        
        # Clean whitespace
        for col in ["parent", "module", "phase", "event"]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Filter for the target step
        mask = (df["module"] == step_name) & (df["phase"] == "run")
        df_step = df[mask].copy()
        
        if df_step.empty:
            return None

        # Identify the last contiguous run
        # We detect a break if the gap between consecutive logs is > 1 second (arbitrary but safe for this workload)
        df_step = df_step.sort_values("timestamp")
        df_step['gap'] = df_step['timestamp'].diff()
        # Find indices where a new run starts (gap > 1s)
        run_starts = df_step[df_step['gap'] > 1.0].index.tolist()
        if run_starts:
            last_run_idx = run_starts[-1]
            df_run = df_step.loc[last_run_idx:].copy()
        else:
            df_run = df_step.copy()

        # Calculate durations: End - Start for each pair
        df_run["prev_timestamp"] = df_run["timestamp"].shift(1)
        df_run["prev_event"] = df_run["event"].shift(1)
        
        durations = df_run[
            (df_run["event"] == "end") & 
            (df_run["prev_event"] == "start")
        ].copy()
        
        latency_ms = (durations["timestamp"] - durations["prev_timestamp"]) * 1000
        
        # Skip warmup (first 5 batches)
        if len(latency_ms) > 5:
            return latency_ms.iloc[5:]
        return latency_ms
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

# Files
baseline_file = "collocation-benchmark/evaluation/results/baseline_finetune.csv"
choreo_file = "collocation-benchmark/evaluation/results/efficientnetv2_s_imagenette_training.csv"

b_latencies = analyze_file(baseline_file, "training_step")
c_latencies = analyze_file(choreo_file, "EfficientNet training")

if b_latencies is not None and c_latencies is not None:
    print(f"\n--- Operational Overhead Analysis (N_baseline={len(b_latencies)}, N_choreo={len(c_latencies)}) ---")
    
    metrics = {
        "Median (ms)": (b_latencies.median(), c_latencies.median()),
        "Mean (ms)": (b_latencies.mean(), c_latencies.mean()),
        "P99 (ms)": (np.percentile(b_latencies, 99), np.percentile(c_latencies, 99)),
        "Std Dev": (b_latencies.std(), c_latencies.std())
    }
    
    print(f"{ 'Metric':<25} | { 'Baseline':<12} | { 'Choreo':<12} | { 'Overhead %':<10}")
    print("-" * 75)
    for name, (b_val, c_val) in metrics.items():
        ovh = ((c_val - b_val) / b_val) * 100
        print(f"{ name:<25} | { b_val:<12.2f} | { c_val:<12.2f} | { ovh:<9.2f}%")
else:
    print("Could not complete analysis due to missing data.")

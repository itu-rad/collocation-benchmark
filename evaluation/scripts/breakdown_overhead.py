import pandas as pd
import os
import numpy as np

def analyze_stage(filepath, stage_name):
    try:
        # timestamp, parent, module, phase, event, ...
        # Column 0: ts, Column 2: module, Column 3: phase, Column 4: event
        df = pd.read_csv(filepath, header=None, on_bad_lines='skip')
        
        # Select and rename
        df = df[[0, 2, 3, 4]].copy()
        df.columns = ["timestamp", "module", "phase", "event"]
        
        # Clean whitespace
        for col in ["module", "phase", "event"]:
            df[col] = df[col].astype(str).str.strip()
        
        mask = (df["module"] == stage_name) & (df["phase"] == "run")
        df_step = df[mask].copy().sort_values("timestamp")
        
        df_step["prev_timestamp"] = df_step["timestamp"].shift(1)
        df_step["prev_event"] = df_step["event"].shift(1)
        
        durations = df_step[
            (df_step["event"] == "end") & (df_step["prev_event"] == "start")
        ].copy()
        
        latency_ms = (durations["timestamp"] - durations["prev_timestamp"]) * 1000
        if len(latency_ms) > 100:
            return latency_ms.iloc[100:]
        return latency_ms
    except Exception as e:
        print(f"Err {stage_name}: {e}")
        return None

choreo_file = "collocation-benchmark/evaluation/results/efficientnetv2_s_imagenette_training.csv"
baseline_file = "collocation-benchmark/evaluation/results/baseline_finetune.csv"

s0_latencies = analyze_stage(choreo_file, "Load Imagenette samples from TorchVision Dataset")
s1_latencies = analyze_stage(choreo_file, "EfficientNet training")
base_latencies = analyze_stage(baseline_file, "training_step")

if s0_latencies is not None and s1_latencies is not None and base_latencies is not None:
    print("\n--- Detailed Stage Breakdown (Steady State) ---")
    print(f"Monolithic Loop Step Mean: {base_latencies.mean():.2f} ms (N={len(base_latencies)})")
    print(f"Choreo Stage 0 (Load) Mean: {s0_latencies.mean():.2f} ms (N={len(s0_latencies)})")
    print(f"Choreo Stage 1 (Comp) Mean: {s1_latencies.mean():.2f} ms (N={len(s1_latencies)})")
    
    print("\n--- Jitter (Std Dev) ---")
    print(f"Monolithic Loop: {base_latencies.std():.4f} ms")
    print(f"Choreo Stage 1: {s1_latencies.std():.4f} ms")
    print(f"Choreo Stage 0: {s0_latencies.std():.4f} ms")
else:
    print("Missing data for breakdown.")

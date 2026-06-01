import pandas as pd
import glob
import os
import re

def analyze_file(filepath):
    try:
        # Read CSV, only reading the first 5 columns which contain the relevant log info
        # timestamp, msg_part1, msg_part2, msg_part3, msg_part4
        df = pd.read_csv(filepath, header=None, names=["timestamp", "col1", "col2", "col3", "col4"], usecols=[0, 1, 2, 3, 4], on_bad_lines='warn')
        
        # Clean columns
        for col in ["col1", "col2", "col3", "col4"]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Filter for relevant rows: Stage execution
        # col2 should be stage name, col3 phase ("run"), col4 event ("start"/"end")
        mask = (df["col3"] == "run") & (df["col2"].str.startswith("Identity Stage"))
        df_stages = df[mask].copy()
        
        if df_stages.empty:
            print(f"Warning: No stage execution logs found in {filepath}")
            return None

        df_stages = df_stages.sort_values("timestamp")
        
        # Calculate deltas
        df_stages["prev_timestamp"] = df_stages["timestamp"].shift(1)
        df_stages["prev_event"] = df_stages["col4"].shift(1)
        df_stages["prev_stage"] = df_stages["col2"].shift(1)
        
        # Transitions: End(Stage N) -> Start(Stage N+1)
        transitions = df_stages[
            (df_stages["col4"] == "start") & 
            (df_stages["prev_event"] == "end")
        ].copy()
        
        def get_id(name):
            try:
                if pd.isna(name): return -1
                return int(name.split(" ")[-1])
            except:
                return -1
                
        transitions["curr_id"] = transitions["col2"].apply(get_id)
        transitions["prev_id"] = transitions["prev_stage"].apply(get_id)
        
        # We only want forward transitions (0->1, 1->2, etc)
        valid_transitions = transitions[transitions["curr_id"] == transitions["prev_id"] + 1]
        
        latencies = valid_transitions["timestamp"] - valid_transitions["prev_timestamp"]
        
        return latencies.describe()
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

# Main
results = {}
files = glob.glob("collocation-benchmark/evaluation/results/noop_chain_depth_*.csv")
for f in files:
    match = re.search(r"depth_(\d+)", f)
    if match:
        depth = int(match.group(1))
        stats = analyze_file(f)
        if stats is not None:
            results[depth] = stats['mean'] * 1e6 # Convert to microseconds

print("\n--- Structural Overhead Results ---")
print("Average Transition Latency (microseconds) vs Pipeline Depth")
print(f"{ 'Depth':<10} | { 'Overhead (us)':<15}")
print("-" * 30)
for depth in sorted(results.keys()):
    print(f"{depth:<10} | {results[depth]:.2f}")

import pandas as pd
import glob
import os
import re

def analyze_file(filepath):
    try:
        df = pd.read_csv(filepath, header=None, names=["timestamp", "col1", "col2", "col3", "col4"], usecols=[0, 1, 2, 3, 4], on_bad_lines='warn')
        
        for col in ["col1", "col2", "col3", "col4"]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Filter for relevant rows: Stage execution
        mask = (df["col3"] == "run") & (df["col2"].str.contains("Stage"))
        df_stages = df[mask].copy()
        
        if df_stages.empty:
            return None

        df_stages = df_stages.sort_values("timestamp")
        
        # Calculate Transition Overhead: Start(N+1) - End(N)
        df_stages["prev_timestamp"] = df_stages["timestamp"].shift(1)
        df_stages["prev_event"] = df_stages["col4"].shift(1)
        df_stages["prev_stage"] = df_stages["col2"].shift(1)
        
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
        
        valid_transitions = transitions[transitions["curr_id"] == transitions["prev_id"] + 1]
        transition_latencies = valid_transitions["timestamp"] - valid_transitions["prev_timestamp"]
        
        # Calculate Stage Duration: End(N) - Start(N)
        # We need to pivot or shift differently. 
        # Since rows are ordered Start, End, Start, End...
        # End rows should have prev_event == "start" and same stage name
        
        durations = df_stages[
            (df_stages["col4"] == "end") & 
            (df_stages["prev_event"] == "start") &
            (df_stages["col2"] == df_stages["prev_stage"])
        ].copy()
        
        stage_latencies = durations["timestamp"] - durations["prev_timestamp"]
        
        return {
            "transition": transition_latencies.describe(),
            "duration": stage_latencies.describe()
        }
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

results = []
files = glob.glob("collocation-benchmark/evaluation/results/noop_chain_depth_10_size_*.csv")

for f in files:
    match = re.search(r"size_(\d+)_mode_(\w+)", f)
    if match:
        size = int(match.group(1))
        mode = match.group(2)
        stats = analyze_file(f)
        if stats is not None:
            res = {
                "Size": size, 
                "Mode": mode, 
                "Transition Overhead": stats["transition"]['mean'] * 1e6,
                "Stage Duration": stats["duration"]['mean'] * 1e6
            }
            results.append(res)

df_res = pd.DataFrame(results)
if not df_res.empty:
    df_res = df_res.sort_values("Size")
    print("\n--- Payload Size Impact Analysis (Depth 10) ---")
    print("Values in microseconds (us)")
    print(df_res.pivot(index="Size", columns="Mode", values=["Transition Overhead", "Stage Duration"]).round(2))
else:
    print("No results found.")

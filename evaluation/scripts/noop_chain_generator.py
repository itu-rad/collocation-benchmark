import yaml
import argparse
import os

def generate_noop_pipeline(depth: int, payload_size: int, mode: str, output_file: str):
    stages = []
    
    # Determine stage component based on mode
    base_component = "stages.stage.Stage"
    if mode == 'copy':
        base_component = "stages.evaluation.copy_stage.CopyStage"
    
    # Create a chain of N stages
    for i in range(depth):
        component = base_component
        config = {}
        
        # Use DataInjector for the first stage if payload_size > 0
        # The injector handles the initial creation, subsequent stages handle passing/copying
        if i == 0 and payload_size > 0:
            component = "stages.evaluation.data_injector.DataInjector"
            config = {"payload_size": payload_size}
            
        stage_config = {
            "name": f"{mode.capitalize()} Stage {i}",
            "id": i,
            "component": component,
            "outputs": [i + 1] if i < depth - 1 else [],
            "polling_policy": "utils.queues.polling.SingleQueuePolicy",
            "disable_logs": False,
            "config": config
        }
        stages.append(stage_config)

    # Pipeline configuration
    pipeline = {
        "name": f"NoOp Chain Depth {depth} Size {payload_size} Mode {mode}",
        "inputs": [0],
        "outputs": [depth - 1],
        "dataset_stage_id": 0,
        "loadgen": {
            "component": "loadgen.OfflineLoadScheduler",
            "queue_depth": 100,
            "max_queries": 10,
            "timeout": 60000,
            "config": {
                "rate": 0
            }
        },
        "stages": stages
    }

    benchmark_config = {
        "name": f"noop_benchmark_depth_{depth}_size_{payload_size}_mode_{mode}",
        "pipelines": [pipeline]
    }

    with open(output_file, 'w') as f:
        yaml.dump(benchmark_config, f, sort_keys=False)
    
    print(f"Generated {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate No-Op pipeline configurations.")
    parser.add_argument("--depths", nargs="+", type=int, default=[10], help="List of pipeline depths to generate.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[0], help="List of payload sizes in bytes.")
    parser.add_argument("--modes", nargs="+", type=str, default=['ref'], choices=['ref', 'copy'], help="List of modes (ref or copy).")
    parser.add_argument("--out-dir", type=str, default="collocation-benchmark/evaluation/configs/noop", help="Output directory for YAML files.")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for depth in args.depths:
        for size in args.sizes:
            for mode in args.modes:
                suffix = f"_depth_{depth}_size_{size}_mode_{mode}"
                filename = os.path.join(args.out_dir, f"noop{suffix}.yml")
                generate_noop_pipeline(depth, size, mode, filename)

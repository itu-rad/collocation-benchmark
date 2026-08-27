import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import time
import logging
import radt

# Trace column layout must match the framework (utils/logger.py): wall-clock
# column 0 (RadT alignment) + monotonic perf_counter_ns trailing column
# (per-step latency). Inlined here so this standalone baseline stays importable
# without the repo's package on sys.path.
PERF_FORMAT = "%(created)f, %(message)s, %(perf)d"


def _install_perf_clock():
    _orig_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = _orig_factory(*args, **kwargs)
        record.perf = time.perf_counter_ns()
        return record

    logging.setLogRecordFactory(_factory)


def setup_logging(label="baseline_finetune"):
    # Use absolute path for log directory (this experiment's own results dir)
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{label}.csv")

    # THE SAME INSTRUMENT AS CHOREO (main.py, the -p 0 path): install the perf
    # clock, one logging.FileHandler with PERF_FORMAT, attached DIRECTLY to the
    # `benchmark` logger. Nothing else.
    #
    # This used to be a QueueHandler feeding a QueueListener on a background
    # thread, fanning out to a FileHandler AND a StreamHandler. Against Choreo
    # that made the two sides differ in three ways at once, not one:
    #
    #   monolith  async emit (put_nowait; format+write+flush happen on the
    #             listener thread), sinks = file + stderr
    #   Choreo    synchronous emit (write+flush syscall on the emitting thread,
    #             under the handler lock), sink = file
    #
    # The emit-path difference sits INSIDE the measured interval on both sides
    # and does not cancel, and the stderr sink made the monolith's cost depend
    # on whether stderr was a terminal or a redirect -- an uncontrolled term in
    # one side only. The control also ran two threads while standing in for the
    # single-threaded floor.
    #
    # Cost of matching: the monolith now pays a synchronous write+flush per row
    # on its training thread instead of a queue put, roughly +15 us/step against
    # effects of 240-2100 us. It also drops a thread.
    #
    # mode='w' is the one deliberate difference from main.py: append mode
    # silently accumulated stale sessions into re-runs, which biased the median.
    _install_perf_clock()
    formatter = logging.Formatter(PERF_FORMAT)
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger, file_handler


def parse_args():
    ap = argparse.ArgumentParser(
        description="Hand-written PyTorch baseline for the modularity-overhead "
        "experiment: a torchvision Imagenette transfer-learning fine-tune, "
        "monolithic (no Choreo framework). Model/weights/batch are parametric so "
        "the scale sweep can drive baseline and Choreo through identical cells."
    )
    ap.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"],
                    default="auto", help="compute device (auto: cuda>mps>cpu)")
    ap.add_argument("--model", default="efficientnet_v2_s",
                    help="torchvision.models factory name (e.g. efficientnet_v2_s, "
                         "efficientnet_v2_m, efficientnet_v2_l). "
                         "MUST match the Choreo cell's model.component so both arms "
                         "do identical per-step work.")
    ap.add_argument("--weights", default="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
                    help="torchvision weights enum name for the preprocessing "
                         "transform (input resolution). MUST match the Choreo cell's "
                         "dataloader dataset.weights.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes (match the Choreo config)")
    ap.add_argument("--max-batches", type=int, default=1000,
                    help="number of training steps to run")
    ap.add_argument("--label", default="baseline_finetune",
                    help="output CSV basename in results/ (so R runs don't collide)")
    ap.add_argument("--no-radt", action="store_true",
                    help="skip the RADTBenchmark telemetry wrapper -> a true "
                         "zero-framework control (used by collect_e2.sh)")
    return ap.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def replace_classifier_head(model, num_classes):
    """Replace the model's final module with Linear(in_features, num_classes).

    Ported VERBATIM from the Choreo stage's transfer-learning setup
    (stages/torchvision_classification/classification.py: named_modules()[-1] +
    _replace_last_module) so the baseline builds the byte-identical head — the
    same last-layer location, in_features, and trainable-param set the wrapped
    arm builds. If these two diverged the per-step overhead would be measuring
    two different workloads. Returns the last module's dotted name so the caller
    selects the same trainable params (`last_name in param_name`) as the stage.
    """
    *_, (last_name, last_mod) = model.named_modules()
    new_head = nn.Linear(last_mod.in_features, num_classes)
    parts = last_name.split(".")
    if len(parts) == 1:
        setattr(model, parts[0], new_head)
    else:
        model.__getattr__(".".join(parts[:-1]))[int(parts[-1])] = new_head
    return last_name


def multi_epoch(loader):
    """Yield batches across as many epochs as needed. Small datasets (e.g.
    Imagenette at large batch -> few batches/epoch: b64 gives only ~147) would
    otherwise stop after one epoch, leaving too few steady-state steps once the
    warm-up is dropped. The Choreo OfflineLoadScheduler already loops epochs
    (its outer `while counter < max_queries`); this makes the baseline match, so
    both arms reach the same step budget at any batch size. The caller bounds it
    via `batch_count >= max_batches`."""
    while True:
        for batch in loader:
            yield batch


def run_training(args, logger):
    batch_size = args.batch_size
    num_classes = 10
    lr = 0.001
    max_batches = args.max_batches

    device = resolve_device(args.device)

    logger.info(f"baseline_finetune, system, setup, start, device={device}")

    # Preprocessing transform (and thus input resolution) comes from the cell's
    # weights enum — the same one the Choreo dataloader resolves via get_weight,
    # so both arms feed the model identically shaped batches.
    weights = models.get_weight(args.weights)
    preprocess = weights.transforms()

    # Same location the Choreo TorchVisionDataLoader uses (cwd/tmp/...), so both
    # implementations share one downloaded copy; auto-download if missing.
    dataset_path = os.path.join(os.getcwd(), "tmp", "torchvision_dataset", "Imagenette")
    os.makedirs(dataset_path, exist_ok=True)
    already = os.path.isdir(os.path.join(dataset_path, "imagenette2"))

    try:
        train_dataset = datasets.Imagenette(
            root=dataset_path,
            split="train",
            size="full",
            download=not already,
            transform=preprocess,
        )
    except Exception as e:
        logger.error(f"baseline_finetune, system, error, {str(e)}")
        return

    # drop_last=True to match TorchVisionDataLoader exactly
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Build from the cell's factory name with random weights (weights=None), then
    # replace the classifier head — mirrors the Choreo stage exactly.
    model = models.get_model(args.model, weights=None)

    # Freezing logic to match Choreo's Transfer Learning: freeze everything, then
    # unfreeze only the replaced head's params (selected by name, like the stage).
    for param in model.parameters():
        param.requires_grad = False

    last_name = replace_classifier_head(model, num_classes)
    params_to_update = []
    for name, param in model.named_parameters():
        if last_name in name:
            param.requires_grad = True
            params_to_update.append(param)
    model = model.to(device)

    optimizer = optim.Adam(params_to_update, lr=lr)
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"baseline_finetune, model, info, trainable_params={trainable_params}, total_params={total_params}"
    )
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {total_params}")

    logger.info("baseline_finetune, system, setup, end")

    batch_count = 0

    logger.info(f"baseline_finetune, training_loop, run, start")

    try:
        # multi_epoch re-iterates the loader across epochs so we always reach
        # max_batches, even when one epoch yields fewer batches (large batch size).
        for inputs, labels in multi_epoch(train_loader):
            if batch_count >= max_batches:
                break

            logger.info("baseline_finetune, training_step, run, start")

            # Match Choreo's per-batch state management
            model.train()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(device)
                # Explicitly cast to LongTensor to match Choreo's classification.py
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Synchronize GPU to ensure timing captures actual hardware execution
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            logger.info("baseline_finetune, training_step, run, end")

            batch_count += 1

    except KeyboardInterrupt:
        logger.info("baseline_finetune, training_loop, run, interrupted")

    logger.info(f"baseline_finetune, training_loop, run, end")


def main():
    args = parse_args()
    logger, file_handler = setup_logging(args.label)

    try:
        if args.no_radt:
            # True zero-framework control: no RadT telemetry/listeners.
            run_training(args, logger)
        else:
            # Wrap execution in RADTBenchmark for hardware metrics
            with radt.run.RADTBenchmark() as run:
                run_training(args, logger)
    finally:
        # Flush and close explicitly: with no listener thread draining on exit,
        # the last rows would otherwise be lost on a hard teardown.
        file_handler.flush()
        file_handler.close()


if __name__ == "__main__":
    main()

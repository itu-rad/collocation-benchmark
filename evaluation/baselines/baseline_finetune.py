import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import time
import os
import logging
import radt

# Global handler for manual flushing
file_handler = None


def setup_logging():
    global file_handler
    # Use absolute path for log directory to avoid relative path issues in radt
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "baseline_finetune.csv")

    formatter = logging.Formatter("%(created)f, %(message)s")
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def flush_logs():
    if file_handler:
        file_handler.flush()
        try:
            os.fsync(file_handler.stream.fileno())
        except:
            pass


def main():
    logger = setup_logging()

    # Wrap execution in RADTBenchmark for hardware metrics
    with radt.run.RADTBenchmark() as run:
        batch_size = 8  # Reduced from 64 to avoid MPS OOM
        num_classes = 10
        lr = 0.001
        max_batches = 100  # Reduced for reliable completion

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger.info(f"baseline_finetune, system, setup, start, device={device}")
        flush_logs()

        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()

        # Path relative to evaluation/baselines
        dataset_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../../tmp/torchvision_dataset/Imagenette"
            )
        )
        os.makedirs(dataset_path, exist_ok=True)

        try:
            try:
                train_dataset = datasets.Imagenette(
                    root=dataset_path,
                    split="train",
                    size="full",
                    download=False,
                    transform=preprocess,
                )
            except RuntimeError:
                train_dataset = datasets.Imagenette(
                    root=dataset_path,
                    split="train",
                    size="full",
                    download=True,
                    transform=preprocess,
                )
        except Exception as e:
            logger.error(f"baseline_finetune, system, error, {str(e)}")
            flush_logs()
            return

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        model = models.efficientnet_v2_s(weights=None)

        # Freezing logic to match Choreo's Transfer Learning
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

        # Optimizer only for trainable parameters
        params_to_update = [p for p in model.parameters() if p.requires_grad]
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
        flush_logs()

        model.train()
        batch_count = 0

        logger.info(f"baseline_finetune, training_loop, run, start")
        flush_logs()

        try:
            for inputs, labels in train_loader:
                if batch_count >= max_batches:
                    break

                logger.info("baseline_finetune, training_step, run, start")
                # Removed flush_logs() here to avoid I/O overhead

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                logger.info("baseline_finetune, training_step, run, end")
                # Removed flush_logs() here to avoid I/O overhead

                batch_count += 1
                # print(f"Batch {batch_count}/{max_batches}")

        except KeyboardInterrupt:
            logger.info("baseline_finetune, training_loop, run, interrupted")
            flush_logs()

        logger.info(f"baseline_finetune, training_loop, run, end")
        flush_logs()


if __name__ == "__main__":
    main()

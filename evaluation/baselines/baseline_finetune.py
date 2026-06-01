import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import logging
from logging.handlers import QueueHandler, QueueListener
import queue
import radt

# Global listener to ensure we can stop it at the end
log_listener = None


def setup_logging():
    global log_listener
    # Use absolute path for log directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "baseline_finetune.csv")

    formatter = logging.Formatter("%(created)f, %(message)s")

    # These handlers will be used by the Listener in a background thread
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Create a queue for the QueueHandler to push to
    log_queue = queue.Queue(-1)
    q_handler = QueueHandler(log_queue)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(q_handler)

    # Start the listener in a background thread to handle the actual I/O
    log_listener = QueueListener(log_queue, file_handler, stream_handler)
    log_listener.start()

    return logger


def main():
    logger = setup_logging()

    # Wrap execution in RADTBenchmark for hardware metrics
    try:
        with radt.run.RADTBenchmark() as run:
            batch_size = 8
            num_classes = 10
            lr = 0.001
            max_batches = 1000

            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            logger.info(f"baseline_finetune, system, setup, start, device={device}")

            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            preprocess = weights.transforms()

            dataset_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../../tmp/torchvision_dataset/Imagenette"
                )
            )
            os.makedirs(dataset_path, exist_ok=True)

            try:
                train_dataset = datasets.Imagenette(
                    root=dataset_path,
                    split="train",
                    size="full",
                    download=False,
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
                num_workers=2,
                drop_last=True,
            )

            model = models.efficientnet_v2_s(weights=None)

            # Freezing logic to match Choreo's Transfer Learning
            for param in model.parameters():
                param.requires_grad = False

            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            model = model.to(device)

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

            batch_count = 0

            logger.info(f"baseline_finetune, training_loop, run, start")

            try:
                for inputs, labels in train_loader:
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

    finally:
        if log_listener:
            log_listener.stop()


if __name__ == "__main__":
    main()

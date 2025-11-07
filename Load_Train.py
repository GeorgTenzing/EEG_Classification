import os 
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import ConfusionMatrix
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from copy import deepcopy

from Models import EEGDataset   
from Utils import plot_training_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')        # for better performance on some hardware
    
import logging
import warnings

# Silence Lightning info messages
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
        
def run_multiple_models(models=None, shared_parameters=None):
    """
    Train multiple EEG models sequentially on the same dataset and dataloaders.
    Loads data and dataloaders once, but creates a new Trainer+Logger for each model.

    Args:
        models (list): List of model classes to train. If None, all predefined models are used.
        shared_parameters (dict, optional): Common parameters such as MAX_TIME, MODEL_KWARGS, etc.

    Returns:
        dict: {model_name: {'best_model': ..., 'metrics_path': ...}}
    """

    
    defaults = {
        "data_path": "datasets/numpy/ssvep_10_nofilter_GCGG.npz",
        
        "MAX_TIME": "00:00:15:00",
        "EPOCHS": 90000,
        "BENCHMARK": True,
        "ACCUM_GRAD_BATCHES": 2,
        "MAX_GRAD_NORM": 1.0,
        "SUMMARY": True,    
        
        "BATCH_SIZE": 256,
        "NUM_WORKERS": 4,
        "PREFETCH_FACTOR": 4,
        "SHUFFLE": True,
        
        "OCCIPITAL_SLICE": slice(0, 8),
        
        "TRAIN_SPLIT": 0.7,
        "VAL_SPLIT": 0.15,
        "TEST_SPLIT": 0.15,
        
        "MODEL_KWARGS": {
            "in_channels": 8,
            "num_classes": 6,
            "LR": 1e-3,
            "WEIGHT_DECAY": 0.0,
        },
    }
        
    params = deepcopy(defaults)
    if shared_parameters:
        params.update(shared_parameters)

    # ============================================================
    # 1️ Load and prepare dataset once
    # ============================================================
    print(f"Loading data from: {params['data_path']}")
    data = np.load(params["data_path"])
    X_all, y_all = data["X"], data["y"]
    print(f"Data loaded: X={X_all.shape}, y={y_all.shape}")

    dataset = EEGDataset(X_all, y_all, occipital_slice=params["OCCIPITAL_SLICE"])

    n_total = len(dataset)
    n_train = int(params["TRAIN_SPLIT"] * n_total)
    n_val   = int(params["VAL_SPLIT"] * n_total)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # ============================================================
    # 2️ Create dataloaders once
    # ============================================================
    train_loader = DataLoader(
        train_ds, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"],
        num_workers=params["NUM_WORKERS"], pin_memory=True, persistent_workers=True,
        prefetch_factor=params["PREFETCH_FACTOR"]
    )
    val_loader  = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print("Dataloaders ready")

    # ============================================================
    # 3️ Define available models if none given
    # ============================================================
    from Models import (
        EEGClassifier, EEGNet, MNISTNet,
    )

    all_models = [
        EEGClassifier, EEGNet, MNISTNet,
    ]

    models_to_run = models or all_models
    # ============================================================
    # 4️ Loop over each model (new Trainer & Logger inside)
    # ============================================================
    results = {}
    for ModelClass in models_to_run:
        model_name = ModelClass.__name__
        print(f"\nTraining {model_name}...\n")

        try:
            # ---- Create model
            model = ModelClass(**params["MODEL_KWARGS"]).to(device)

            # ---- Logger and checkpoint (unique per model)
            csv_logger = CSVLogger("logs", name=model_name)
            checkpoint = ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1,
                filename="best-{epoch:02d}-val_acc={val_acc:.3f}",
                auto_insert_metric_name=False,
            )

            # ---- Fresh trainer for this model
            trainer = Trainer(
                max_time=params["MAX_TIME"],
                max_epochs=params["EPOCHS"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision=16,
                gradient_clip_val=params["MAX_GRAD_NORM"],
                accumulate_grad_batches=params["ACCUM_GRAD_BATCHES"],

                enable_model_summary=params["SUMMARY"],
                enable_checkpointing=True,
                callbacks=[checkpoint],
                benchmark=params["BENCHMARK"],
                fast_dev_run=False,
                
                logger=csv_logger,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
                enable_progress_bar=True,
            )

            # ---- Train
            trainer.fit(model, train_loader, val_loader)

            # ---- Load best model
            best_path = trainer.checkpoint_callback.best_model_path
            print(f"Best model saved at: {best_path}")
            best_model = ModelClass.load_from_checkpoint(best_path, **params["MODEL_KWARGS"])

            log_dir = os.path.dirname(os.path.dirname(best_path))
            metrics_path = os.path.join(log_dir, "metrics.csv")

            results[model_name] = {
                "best_model": best_model,
                "metrics_path": metrics_path
            }

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            traceback.print_exc()
            continue

    print("\nAll models processed successfully!\n")
    return results, test_loader


        
def test_trained_models(results, test_loader):
    """
    Evaluate trained models returned by run_multiple_models() on the shared test_loader.

    Args:
        results (dict): From run_multiple_models(), containing
                        {'best_model': model, 'metrics_path': ...} per model.
        test_loader (DataLoader): Shared test set.

    Returns:
        dict: {model_name: {'test_acc': float, 'confusion_matrix': np.ndarray}}
    """

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")

    results_out = {}
    for name, info in results.items():
        print(f"\nTesting {name}...")

        try:
            model = info["best_model"].to(device)
            model.eval()

            # ---------------------------------------------------
            # Lightning test accuracy (if test_step defined)
            # ---------------------------------------------------
            metrics = trainer.test(model, dataloaders=test_loader, verbose=False)[0]
            acc = float(metrics.get("test_acc", 0.0))
            
            # ---------------------------------------------------
            # Compute confusion matrix manually
            # ---------------------------------------------------
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X, y in test_loader:
                    preds = model(X)
                    preds = torch.argmax(preds, dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(y.cpu())

            all_preds   = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

            # Use torchmetrics for a consistent CM
            cm_metric = ConfusionMatrix(task="multiclass", num_classes=int(torch.max(all_targets)) + 1)
            cm = cm_metric(all_preds, all_targets).numpy()

            results_out[name] = {
                "test_acc": acc,
                "confusion_matrix": cm
            }

            print(f"{name}: Test accuracy = {acc:.3f}")

        except Exception as e:
            print(f"Error testing {name}: {e}")
            results_out[name] = None
            continue

    print("\nTesting complete for all models.\n")
    return results_out


def plot_all_results(results, test_results=None):
    for name, info in results.items():
        print(f"\nPlotting {name}: Test Accuracy = {test_results[name]['test_acc']:.3f}")
        plot_training_metrics(info["metrics_path"])
        

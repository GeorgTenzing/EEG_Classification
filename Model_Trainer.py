import os 
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import ConfusionMatrix
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from copy import deepcopy

from Dataset_torch import EEGDataset_with_filters
from Models_1D import EEGClassifier 
from Utils import plot_training_metrics, benchmark_loader, epoch_time

device = "cuda" if torch.cuda.is_available() else "cpu"
    
import logging
import warnings

# Silence Lightning info messages
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ideas: Gradient clipping, label smoothing
   
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
        "DATASET_CLASS": EEGDataset_with_filters,
        "NOTCH_50": False,
        "SAMPLE_RATE": 500,
        "RUNS": [4, 8, 12],
        "CLASSES": ["T0", "T1", "T2"],
        
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
        "SUBJECTS": [s for s in range(1, 109) if s not in [88, 92, 100, 104]],
        
        "TRAIN_SPLIT": 0.7,
        "VAL_SPLIT": 0.15,
        "TEST_SPLIT": 0.15,
        
        "MODEL_KWARGS": {
            "in_channels": 8,
            "num_classes": 6,
            "LR": 1e-3,
            "WEIGHT_DECAY": 0.0,
            "class_labels": [0, 7, 10.5, 12, 15.2, 18.1],
            "class_weights": None,
        },
        "testing": True,
        "LOAD_CHECKPOINT": None,
        "skip_training": False,
    }
        
    params = deepcopy(defaults)
    if shared_parameters:
        params.update(shared_parameters)  

    # ============================================================
    # 1️ Load, prepare and split datasets
    # ============================================================
    print(f"Loading data from: {params['data_path']}")
    if params["data_path"].endswith(".npz"):
        data = np.load(params["data_path"])
        X_all, y_all = data["X"], data["y"]
        print(f"Data loaded: X={X_all.shape}, y={y_all.shape}")
        
        dataset = params["DATASET_CLASS"](
            X_all, 
            y_all, 
            occipital_slice=params["OCCIPITAL_SLICE"], 
            notch_50=params["NOTCH_50"], 
            sample_rate=params["SAMPLE_RATE"],
        )
    else:
        dataset = params["DATASET_CLASS"](
            data_path=params["data_path"],
            occipital_slice=params["OCCIPITAL_SLICE"], 
            notch_50=params["NOTCH_50"], 
            sample_rate=params["SAMPLE_RATE"],
            subjects=params["SUBJECTS"],
            runs=params["RUNS"],
            classes=params["CLASSES"],
        )

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
    # 2️ Create dataloaders for each split
    # ============================================================
    train_loader = DataLoader(
        train_ds, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"],
        num_workers=params["NUM_WORKERS"], pin_memory=True, persistent_workers=True,
        prefetch_factor=params["PREFETCH_FACTOR"]
    )
    val_loader  = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print("Dataloaders ready")
    # benchmark_loader(train_ds, batch_size=params["BATCH_SIZE"])
    epoch_time(train_loader, batch_size=params["BATCH_SIZE"])
    
    # ============================================================
    # 3️ Loop over each model (new Trainer & Logger inside)
    # ============================================================
    models_to_run = models or [EEGClassifier]
    results = {}
    for ModelClass in models_to_run:
        model_name = ModelClass.__name__
        print(f"\nProcessing  {model_name}...\n")

        try: 
            
            checkpoint = ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1,
                filename="best-{epoch:02d}-val_acc={val_acc:.3f}",
                auto_insert_metric_name=False,
            )
            trainer = Trainer(
                max_time=params["MAX_TIME"],
                max_epochs=params["EPOCHS"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision=16,
                gradient_clip_val=params["MAX_GRAD_NORM"],
                accumulate_grad_batches=params["ACCUM_GRAD_BATCHES"],

                enable_model_summary=params["SUMMARY"],
                enable_checkpointing=False if params["skip_training"] else True,
                callbacks=None if params["skip_training"] else [checkpoint],
                benchmark=params["BENCHMARK"],
                fast_dev_run=False,
                
                logger=None if params["skip_training"] else CSVLogger("logs", name=model_name),
                num_sanity_val_steps=0,
                log_every_n_steps=0,
                enable_progress_bar=False,
                
                # detect_anomaly=True,
            )
            
            # =====================================================
            # OPTION 1: LOAD CHECKPOINT AND SKIP TRAINING
            # =====================================================
            if params["skip_training"]:
                best_path = params["LOAD_CHECKPOINT"]
                print(f"Loading model from checkpoint instead of training:\n  {best_path}")
              
            # =====================================================
            # OPTION 2: TRAIN MODEL FROM CHECKPOINT OR FROM SCRATCH
            # =====================================================      
            else:
                print(f"Training {model_name}...\n")
                
                # ---- Load from checkpoint and continue training ----
                if params["LOAD_CHECKPOINT"] is not None:
                    model = ModelClass.load_from_checkpoint(
                        params["LOAD_CHECKPOINT"], 
                        **params["MODEL_KWARGS"]
                    ).to(device)  # Lightning will load params into it
                else:
                    # ---- Create model ----
                    model = ModelClass(**params["MODEL_KWARGS"]).to(device)
                
                if params["SUMMARY"]:  
                    print(ModelSummary(model, max_depth=1))
                # ---- Train ----
                trainer.fit(model, train_loader, val_loader)

                # ---- Load best model ----
                best_path = trainer.checkpoint_callback.best_model_path
                print(f"Best model saved at: {best_path}")
                
        
            # ----------------------------------------------------
            # Load best model for testing and metrics plotting
            # ----------------------------------------------------
            best_model = ModelClass.load_from_checkpoint(
                    best_path,
                    **params["MODEL_KWARGS"]
                ).to(device)
                
            log_dir = os.path.dirname(os.path.dirname(best_path))
            metrics_path = os.path.join(log_dir, "metrics.csv")
            results[model_name] = {
                "best_model": best_model,
                "metrics_path": metrics_path
            }
            
            # =====================================================
            # OPTION 4: Testing phase (if enabled)
            # =====================================================
            if params["testing"]:
                if params["SUMMARY"]:  
                    print(ModelSummary(best_model, max_depth=1))
                
                best_model.eval()
                metrics = trainer.test(best_model, dataloaders=test_loader, verbose=False)[0]
                acc = float(metrics.get("test_acc", 0.0))
                print(f"{model_name}: Test accuracy = {acc:.3f}")
                
                # ---------------------------------------------------
                # Compute confusion matrix manually
                # ---------------------------------------------------
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for X, y in test_loader:
                        preds = best_model(X)
                        preds = torch.argmax(preds, dim=1)
                        all_preds.append(preds.cpu())
                        all_targets.append(y.cpu())
                all_preds   = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)

                # Use torchmetrics for a consistent CM
                cm_metric = ConfusionMatrix(task="multiclass", num_classes=int(torch.max(all_targets)) + 1)
                cm = cm_metric(all_preds, all_targets).numpy()
            
                # Plot training metrics
                for name, info in results.items():
                    print(f"\nPlotting {name}: Test Accuracy = {acc:.3f}")
                    plot_training_metrics(info["metrics_path"])
            
        

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            traceback.print_exc()
            continue

    print("\nAll models processed successfully!\n")
    return results, test_loader


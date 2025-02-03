# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A starter script for fine-tuning a BERT model on your own dataset."""

import os
import sys
from typing import Optional, cast

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import src.evals.data as data_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import src.flex_bert as flex_bert_module
from src.hf_bert import CustomBertEmbeddings
import transformers
from composer import Trainer, algorithms, Evaluator
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.core.types import Dataset
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from src.scheduler import WarmupStableDecayScheduler
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
import torch
import pandas as pd
from transformers import BertConfig

# Import the synthetic data generation function
from generate_dataset import generate_synthetic_dataset, create_extended_letter_mapping, transform_discrete_to_letters

def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f"WARNING: device_train_microbatch_size > device_train_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_train_batch_size}."
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size

    # Safely set `device_eval_microbatch_size` if not provided by user
    if "device_eval_microbatch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_microbatch_size = 1
        else:
            cfg.device_eval_microbatch_size = cfg.device_train_microbatch_size

    global_eval_batch_size, device_eval_microbatch_size = (
        cfg.get("global_eval_batch_size", global_batch_size),
        cfg.device_eval_microbatch_size,
    )
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()
    if isinstance(device_eval_microbatch_size, int):
        if device_eval_microbatch_size > device_eval_microbatch_size:
            print(
                f"WARNING: device_eval_microbatch_size > device_eval_batch_size, "
                f"will be reduced from {device_eval_microbatch_size} -> {device_eval_batch_size}."
            )
            device_eval_microbatch_size = device_eval_batch_size
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if "wandb" in cfg.get("loggers", {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def build_callback(name, kwargs):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1), gpu_flops_available=kwargs.get("gpu_flops_available", None)
        )
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "optimizer_monitor":
        return OptimizerMonitor(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
        )
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name, kwargs):
    if name == "wandb":
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(cfg):
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg, model):
    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def build_my_dataloader(cfg: DictConfig, device_batch_size: int, decimal_points: int = 0, save_data: bool = False):
    """Create a dataloader for classification using synthetically generated data.

    Args:
        cfg (DictConfig): An omegaconf config that houses all the configuration
            variables needed to instruct dataset/dataloader creation.
        device_batch_size (int): The size of the batches that the dataloader
            should produce.
        decimal_points (int): Number of decimal points to keep when converting to strings.

    Returns:
        dataloader: A dataloader set up for use of the Composer Trainer.
    """
    # Generate the synthetic dataset
    df = generate_synthetic_dataset(
        n_samples=cfg.get("n_samples", 100),
        n_continuous_features=cfg.get("n_continuous_features", 15),
        n_discrete_features=cfg.get("n_discrete_features", 15),
        n_classes=cfg.get("n_classes", 2),
        class_distribution=cfg.get("class_distribution", [0.8, 0.2]),
        n_bins=cfg.get("n_bins", 10),
        n_redundant=cfg.get("n_redundant", 5),
        n_noisy=cfg.get("n_noisy", 20),
        class_sep=cfg.get("class_sep", 0.1),
        textual_discrete=cfg.get("textual_discrete", False),
    )

    # Define the label column
    label_column = 'label'

    print(df.shape)

    # Create extended letter mapping for all columns except the label column
    letter_mapping = create_extended_letter_mapping(df, label_column)

    # Transform all columns to letter encoding except the label column
    df_transformed = transform_discrete_to_letters(df.drop(columns=[label_column]), letter_mapping)

    # Add the label column back to the transformed dataframe
    df_transformed[label_column] = df[label_column]

    # Change structure to "sentence", "label" and "idx"
    # All columns except the last one are features and they are concatenated to form a sentence
    # The last column is the label
    df_transformed['sentence'] = df_transformed.drop(columns=[label_column]).apply(lambda x: ' '.join([str(val) for val in x]), axis=1)
    # Remove spaces from the sentence
    df_transformed['sentence'] = df_transformed['sentence'].str.replace(' ', '')

    # Put sentenct equal a if the label is 0 and b if the label is 1
    #df_transformed['sentence'] = df_transformed.apply(lambda x: 'Abrakadabra' if x[label_column] == 0 else 'Abrakababra', axis=1)

    # Reorder columns and add index
    df = df_transformed[['sentence', label_column]]
    df['idx'] = df_transformed.index
    
    #df = df[['sentence', 'label']]
    #df['idx'] = df.index
    
    # Tokenize the dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    tokenized_dataset = tokenizer(
        df['sentence'].tolist(),  # Ensure this is a list of strings
        padding=True,
        truncation=True,
        max_length=cfg.max_seq_len,
        return_tensors='pt'
    )

    # Print input sentence and tokenization results
    print("\nTokenizer Debug Info:")
    print("-" * 50)
    # Print first 3 examples
    for i in range(min(3, len(df))):
        print(f"\nExample {i+1}:")
        print(f"Input sentence: {df['sentence'].iloc[i]}")
        print(f"Label: {df['label'].iloc[i]}")
        
        # Get tokenized ids for this example
        tokens = tokenizer.encode(df['sentence'].iloc[i])
        print(f"Token IDs: {tokens}")
        
        # Decode back to string to verify tokenization
        decoded = tokenizer.decode(tokens)
        print(f"Decoded text: {decoded}")
        
        # Print individual tokens
        tokens_list = tokenizer.convert_ids_to_tokens(tokens)
        print(f"Individual tokens: {tokens_list}")
    print("-" * 50)

    # Create a PyTorch dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    labels = df['label'].tolist()
    custom_dataset = CustomDataset(tokenized_dataset, labels)

    # Create a DataLoader
    dataloader = DataLoader(
        custom_dataset,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(custom_dataset, drop_last=cfg.drop_last, shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )

    # Save the data if requested
    if save_data:
        # Save the dataset
        df.to_csv('synthetic_dataset.csv', index=False)

    return dataloader

from transformers import BertTokenizerFast
def get_tokenizer(tokenizer_name: str):
    # Initialize the tokenizer with whole word masking
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, do_basic_tokenize=False)
    return tokenizer

def build_model(cfg: DictConfig):
    # Note: cfg.num_labels should match the number of classes in your dataset!
    if cfg.name == "hf_bert":
        model = hf_bert_module.create_hf_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.get("pretrained_model_name", "bert-base-uncased"),
            use_pretrained=False,
            model_config=cfg.get("model_config"),
            tokenizer_name=False,
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")
    return model

def train(cfg: DictConfig, return_trainer: bool = False, do_train: bool = True) -> Optional[Trainer]:
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print("Initializing model...")
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params=:.4e}")

    # Dataloaders
    print("Building train loader...")
    train_loader = build_my_dataloader(
        cfg.train_loader,
        cfg.global_train_batch_size // dist.get_world_size(),
        save_data=True
    )

    print("Building eval loader...")
    global_eval_batch_size = cfg.get("global_eval_batch_size", cfg.global_train_batch_size)
    eval_loader = build_my_dataloader(
        cfg.eval_loader,
        cfg.get("device_eval_batch_size", global_eval_batch_size // dist.get_world_size()),
    )
    eval_evaluator = Evaluator(
        label="eval",
        dataloader=eval_loader,
        device_eval_microbatch_size=cfg.get("device_eval_microbatch_size", None),
    )

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get("callbacks", {}).items()]

    # Algorithms
    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get("algorithms", {}).items()]

    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "sequence-classification")

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_evaluator,
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get("device"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=cfg.get("save_folder"),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path"),
        load_weights_only=True,
    )

    print("Logging config...")
    log_config(cfg)

    if do_train:
        print("Starting training...")
        trainer.fit()

    if return_trainer:
        return trainer


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open("yamls/defaults.yaml") as f:
        default_cfg = om.load(f)
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(default_cfg, yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    train(cfg)

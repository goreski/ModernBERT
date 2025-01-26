import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import wandb
from generate_dataset import generate_synthetic_dataset
from composer import Trainer
from composer.models import HuggingFaceModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the neural network model
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_nn(config, train_loader, val_loader):
    input_dim = config['n_continuous_features'] + config['n_discrete_features'] + config['n_redundant'] + config['n_noisy']
    hidden_dim = 64
    output_dim = config['n_classes']

    model = TwoLayerNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_auc": val_auc
        })
        
        print(f"Epoch {epoch + 1}/20, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

def train_xgboost(config, train_df, val_df):
    train_X = train_df.drop(columns=['label'])
    train_y = train_df['label']
    val_X = val_df.drop(columns=['label'])
    val_y = val_df['label']

    model = xgb.XGBClassifier()
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=True)

    val_preds = model.predict(val_X)
    val_accuracy = accuracy_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds, average='weighted')
    val_auc = roc_auc_score(val_y, val_preds, multi_class='ovr')

    # Log metrics to wandb
    wandb.log({
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_auc": val_auc
    })

    print(f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

def train_modernbert(config, train_df, val_df):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    model = AutoModelForSequenceClassification.from_pretrained(config['tokenizer_name'], num_labels=config['num_labels'])

    train_texts = train_df.drop(columns=['label']).values.tolist()
    train_labels = train_df['label'].values.tolist()
    val_texts = val_df.drop(columns=['label']).values.tolist()
    val_labels = val_df['label'].values.tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config['max_seq_len'])
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=config['max_seq_len'])

    train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']), torch.tensor(val_labels))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    composer_model = HuggingFaceModel(model=model, tokenizer=tokenizer)

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        max_duration=config['max_duration'],
        eval_interval=config['eval_interval'],
        optimizers=optim.AdamW(composer_model.parameters(), lr=config['lr']),
        schedulers=[{
            'name': 'linear_decay_with_warmup',
            't_warmup': config['t_warmup'],
            'alpha_f': config['alpha_f']
        }],
        loggers=[wandb]
    )

    trainer.fit()

def main():
    # Initialize wandb
    wandb.init(project="model_comparison")

    # Load configuration
    config = wandb.config

    # Generate the dataset
    df = generate_synthetic_dataset(
        n_samples=config.n_samples,
        n_continuous_features=config.n_continuous_features,
        n_discrete_features=config.n_discrete_features,
        n_classes=config.n_classes,
        class_distribution=config.class_distribution,
        n_bins=config.n_bins,
        n_redundant=config.n_redundant,
        n_noisy=config.n_noisy,
        class_sep=config.class_sep
    )

    # Split the dataset into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.33, stratify=temp_df['label'], random_state=42)

    # Convert DataFrames to PyTorch tensors
    def df_to_tensor(df):
        X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
        y = torch.tensor(df['label'].values, dtype=torch.long)
        return TensorDataset(X, y)

    train_dataset = df_to_tensor(train_df)
    val_dataset = df_to_tensor(val_df)
    test_dataset = df_to_tensor(test_df)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train models
    if config.model_name == "nn":
        train_nn(config, train_loader, val_loader)
    elif config.model_name == "xgboost":
        train_xgboost(config, train_df, val_df)
    elif config.model_name == "modernbert":
        from omegaconf import OmegaConf as om
        from omegaconf import DictConfig
        from typing import Optional, cast

        yaml_path = "yamls/test/sequence_classification.yaml" 
        with open("yamls/defaults.yaml") as f:
            default_cfg = om.load(f)
        with open(yaml_path) as f:
            yaml_cfg = om.load(f)
        cfg = om.merge(default_cfg, yaml_cfg)
        cfg = cast(DictConfig, cfg)  # for type checking
        print(cfg)
        train_modernbert(cfg, train_df, val_df)

if __name__ == "__main__":
    main()

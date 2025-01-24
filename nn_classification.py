import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Import the dataset generation function
from generate_dataset import generate_synthetic_dataset

# Initialize wandb
wandb.init(project="nn_classification")

# Generate the dataset
n_samples = 1000
n_continuous_features = 15
n_discrete_features = 15
n_classes = 2
class_distribution = [0.8, 0.2]
n_bins = 10
n_redundant = 5
n_noisy = 20
class_sep = 0.1

df = generate_synthetic_dataset(n_samples, n_continuous_features, n_discrete_features, n_classes, class_distribution, n_bins, n_redundant, n_noisy, class_sep)

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

input_dim = n_continuous_features + n_discrete_features + n_redundant + n_noisy
hidden_dim = 64
output_dim = n_classes

model = TwoLayerNN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
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
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Evaluate the model on the test set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_accuracy = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='weighted')
test_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')

# Log test metrics to wandb
wandb.log({
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
    "test_auc": test_auc
})

print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")

import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import time
import math

from pathlib import Path
from torch import nn
from processing import process_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3, TriBCE_Loss, Weighted_TriBCE_Loss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

model_save_dir = os.path.join(Path().resolve(), "model/music")
os.makedirs(model_save_dir, exist_ok=True)

def print_class_distribution(y_data, name):
    total = len(y_data)
    class_counts = y_data.value_counts().to_dict()
    print(f"\n{name} Data - Total Samples: {total}")
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        percent = (count / total) * 100
        print(f"  Class {label}: {count} samples ({percent:.2f}%)")

def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, patience=3):
    train_losses = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_scores = []

    project_root = Path().resolve()
    model_path = "model/music/best_model.pth"
    model_path = os.path.join(project_root, model_path)

    best_val_loss = float('inf')  
    if os.path.exists(model_path):
        previous_model = torch.load(model_path, map_location=device)
        best_previous_loss = previous_model['loss']
    else:
        best_previous_loss = float('inf')
    early_stop_counter = 0       

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_y_true = []
        all_y_scores = []

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output['y_pred'], labels, output['y_d'], output['y_s'])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            all_y_true.extend(labels.detach().cpu().numpy())
            all_y_scores.extend(output['y_pred'].detach().cpu().numpy())

        all_y_true = np.array(all_y_true).flatten()
        all_y_scores = np.array(all_y_scores).flatten()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        y_pred = (all_y_scores > 0.5).astype(int)

        precision = precision_score(all_y_true, y_pred, zero_division=0, pos_label=1)
        recall = recall_score(all_y_true, y_pred, zero_division=0, pos_label=1)
        f1 = f1_score(all_y_true, y_pred, zero_division=0, pos_label=1)
        auc = roc_auc_score(all_y_true, all_y_scores)
        accuracy = accuracy_score(all_y_true, all_y_scores)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

        print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC: {auc:.4f}')

        val_loss = validate(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            if best_val_loss < best_previous_loss:
                save_checkpoint(model, epoch, optimizer, best_val_loss, path="model/music/best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    all_y_true = []
    all_y_scores = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = loss_fn(output['y_pred'], labels, output['y_d'], output['y_s'])
            val_loss += loss.item()

            all_y_true.extend(labels.detach().cpu().numpy())
            all_y_scores.extend(output['y_pred'].detach().cpu().numpy())

    all_y_true = np.array(all_y_true).flatten()
    all_y_scores = np.array(all_y_scores).flatten()

    avg_val_loss = val_loss / len(val_loader)

    y_pred = (all_y_scores > 0.5).astype(int)

    precision = precision_score(all_y_true, y_pred, zero_division=0)
    recall = recall_score(all_y_true, y_pred, zero_division=0)
    f1 = f1_score(all_y_true, y_pred, zero_division=0)
    auc = roc_auc_score(all_y_true, all_y_scores)
    accuracy = accuracy_score(all_y_true, all_y_scores)

    print(f'\nValidation Loss: {avg_val_loss:.4f}')
    print(f'\nAccuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}\n')

    model.train()  
    return avg_val_loss  

def save_checkpoint(model, epoch, optimizer, loss, path="model_checkpoint.pth"):
    project_root = Path().resolve()
    model_path = os.path.join(project_root, path)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, model_path)
    print(f"Checkpoint saved at epoch {epoch}")

if __name__ == "__main__":
    project_root = Path().resolve()
    data_path = "music/train_data/merged_user_item_duration.parquet"
    data_path = os.path.join(project_root, data_path)
    if os.path.exists(data_path):
        data = pd.read_parquet(data_path)
    else:
        data = process_data(data_path)

    train_data = data.drop(columns=["username", "content_id", "profile_id"])
    train_data = train_data.fillna(0)
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    train_data = train_data.dropna()
    y = train_data["label"]
    X = train_data.drop(columns=["label"])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    raw_root = (len(y) - y.sum()) / y.sum()
    pos_weight = math.sqrt(raw_root)

    print_class_distribution(pd.Series(y_train), "Training")
    print_class_distribution(pd.Series(y_val), "Validation")

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    input_dim = X_train_tensor.shape[1]
    model = DCNv3(input_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = Weighted_TriBCE_Loss(pos_weight=pos_weight)

    start_time = time.time()
    for i in range(5):
        print(f"Training run number: {i+1}")
        epochs = 20
        patience = 3

        train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, patience=patience)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Elapsed time: {elapsed}')
# gnn.py
# Graph Neural Network for relational anomaly detection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import os
import csv

class GNNModel(nn.Module):
    """GNN for relational structure detection."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return x

def build_graph_from_correlation(X_train, threshold=0.5):
    """Build graph structure from feature correlations."""
    corr_matrix = np.corrcoef(X_train.T)
    edges = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def compute_graph_stats(edge_index, num_nodes):
    """Compute graph statistics and save to CSV."""
    ei = edge_index.numpy()
    undirected = set()
    for i in range(ei.shape[1]):
        a, b = ei[0, i], ei[1, i]
        undirected.add((min(a, b), max(a, b)))
    num_edges = len(undirected)
    max_edges = num_nodes * (num_nodes - 1) / 2
    edge_density = num_edges / max_edges if max_edges > 0 else 0.0

    degree = np.zeros(num_nodes, dtype=int)
    for a, b in undirected:
        degree[a] += 1
        degree[b] += 1
    isolated = int(np.sum(degree == 0))

    stats = {
        "nodes": num_nodes,
        "edges": num_edges,
        "edge_density": round(edge_density, 6),
        "mean_degree": round(float(degree.mean()), 4),
        "max_degree": int(degree.max()),
        "min_degree": int(degree.min()),
        "isolated_nodes": isolated,
    }

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/gnn_graph_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic", "value"])
        for k, v in stats.items():
            writer.writerow([k, v])
    return stats

def train_gnn(X_train, y_train, X_val, y_val, edge_index,
              epochs=100, patience=10, seed=42):
    """Train GNN with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = Data(
        x=torch.tensor(X_train, dtype=torch.float32),
        y=torch.tensor(y_train, dtype=torch.long),
        edge_index=edge_index
    )
    val_data = Data(
        x=torch.tensor(X_val, dtype=torch.float32),
        y=torch.tensor(y_val, dtype=torch.long),
        edge_index=edge_index
    )

    model = GNNModel(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            out = model(val_data.x, val_data.edge_index)
            val_loss = criterion(out, val_data.y)
            val_losses.append(val_loss.item())

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save training log
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/gnn_training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i in range(len(train_losses)):
            writer.writerow([i, train_losses[i], val_losses[i]])

    # Save model at best epoch
    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), "results/models/gnn.pt")

    print(f"  Best epoch: {best_epoch} with Val Loss = {best_val_loss:.4f}")
    return model, train_losses, val_losses, best_epoch, best_val_loss

if __name__ == "__main__":
    print("=" * 60)
    print("GNN Training")
    print("=" * 60)

    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")

    print("Building graph structure...")
    edge_index = build_graph_from_correlation(X_train, threshold=0.5)
    print(f"   Graph has {edge_index.shape[1]} directed edges")

    stats = compute_graph_stats(edge_index, num_nodes=X_train.shape[1])
    print(f"   Graph statistics: {stats}")

    model, train_losses, val_losses, best_epoch, best_val_loss = train_gnn(
        X_train, y_train, X_val, y_val, edge_index,
        epochs=100, patience=10, seed=42
    )

    print("GNN training complete!")
    print("=" * 60)
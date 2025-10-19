import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import numpy as np
import pandas as pd
from data_clean import clean_data
from utils import get_project_paths


class GNNVotingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.3):
        super(GNNVotingModel, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(GCNConv(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)


def run_gnn_analysis(hidden_dim=32, num_layers=2, epochs=10000, lr=0.0001, sample_size=None, print_every=100):
    data = clean_data()
    paths = get_project_paths()

    treatment_vars = ['treatment_civic duty', 'treatment_hawthorne', 'treatment_neighbors', 'treatment_self']
    control_vars = ['sex', 'yob', 'g2000', 'g2002', 'p2004', 'p2000', 'p2002']

    feature_cols = treatment_vars + control_vars
    feature_cols = [col for col in feature_cols if col in data.columns]

    if 'zip' not in data.columns or 'plus4' not in data.columns:
        raise ValueError("Need 'zip' and 'plus4' columns for neighborhood definition")

    data = data.dropna(subset=feature_cols + ['voted', 'zip', 'plus4'])

    data['zip_plus4'] = data['zip'].astype(str).str.zfill(5) + '-' + data['plus4'].astype(str).str.zfill(4)

    zip_counts = data['zip_plus4'].value_counts()
    valid_zips = zip_counts[zip_counts > 1].index
    data = data[data['zip_plus4'].isin(valid_zips)]

    if len(data) == 0:
        raise ValueError("No ZIP+4 codes with multiple households")

    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)

    data['node_idx'] = range(len(data))

    edge_list = []
    for zip_plus4_id, zip_group in data.groupby('zip_plus4'):
        nodes_in_zip = zip_group['node_idx'].values
        for i, node_i in enumerate(nodes_in_zip):
            for node_j in nodes_in_zip[i + 1:]:
                edge_list.append([node_i, node_j])
                edge_list.append([node_j, node_i])

    if len(edge_list) == 0:
        raise ValueError("No edges created in graph")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    X_data = data[feature_cols].values
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X_data - X_mean) / X_std

    x = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(data['voted'].values, dtype=torch.float).unsqueeze(1)

    num_nodes = len(data)
    indices_array = np.arange(num_nodes)
    train_idx, val_idx, test_idx = np.split(
        indices_array[np.random.RandomState(42).permutation(num_nodes)],
        [int(0.7 * num_nodes), int(0.85 * num_nodes)]
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    model = GNNVotingModel(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=200)
    criterion = nn.BCELoss()

    best_val_auc = 0
    patience_counter = 0
    patience = 1000

    print(f"Starting training: {num_nodes} nodes, {edge_index.shape[1] // 2} edges")
    print(f"Epochs: {epochs}, LR: {lr}, Hidden: {hidden_dim}")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % print_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(graph_data.x, graph_data.edge_index)
                val_loss = criterion(val_out[graph_data.val_mask], graph_data.y[graph_data.val_mask])
                val_preds = val_out[graph_data.val_mask].numpy().flatten()
                val_labels = graph_data.y[graph_data.val_mask].numpy().flatten()
                val_auc = roc_auc_score(val_labels, val_preds)

                train_preds = out[graph_data.train_mask].detach().numpy().flatten()
                train_labels = graph_data.y[graph_data.train_mask].numpy().flatten()
                train_auc = roc_auc_score(train_labels, train_preds)

                print(f"Epoch {epoch + 1:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
                      f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

                scheduler.step(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f">>> New best validation AUC: {best_val_auc:.4f}")
                else:
                    patience_counter += print_every

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        predictions = model(graph_data.x, graph_data.edge_index)

        train_preds = predictions[graph_data.train_mask].numpy().flatten()
        train_labels = graph_data.y[graph_data.train_mask].numpy().flatten()
        train_preds_binary = (train_preds > 0.5).astype(int)

        test_preds = predictions[graph_data.test_mask].numpy().flatten()
        test_labels = graph_data.y[graph_data.test_mask].numpy().flatten()
        test_preds_binary = (test_preds > 0.5).astype(int)

        train_accuracy = accuracy_score(train_labels, train_preds_binary)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_logloss = log_loss(train_labels, train_preds)

        test_accuracy = accuracy_score(test_labels, test_preds_binary)
        test_auc = roc_auc_score(test_labels, test_preds)
        test_logloss = log_loss(test_labels, test_preds)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Log Loss: {test_logloss:.4f}")
    print("=" * 80)

    results = {
        'model': model,
        'graph_data': graph_data,
        'feature_names': feature_cols,
        'train_accuracy': train_accuracy,
        'train_auc': train_auc,
        'train_logloss': train_logloss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'test_logloss': test_logloss,
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1] // 2,
        'num_features': len(feature_cols),
        'avg_neighbors': edge_index.shape[1] / num_nodes,
        'epochs_trained': epoch + 1,
        'best_val_auc': best_val_auc
    }

    return results

print(run_gnn_analysis())
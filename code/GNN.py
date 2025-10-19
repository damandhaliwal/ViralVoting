import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from data_clean import clean_data
from utils import get_project_paths


class GNNVotingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super(GNNVotingModel, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)


def run_gnn_analysis(hidden_dim=64, num_layers=2, epochs=200, lr=0.01, k_neighbors=8, feature_selection_threshold=0.01):
    data = clean_data()
    paths = get_project_paths()

    exclude_cols = ['voted', 'treatment', 'hh_id', 'cluster', 'zip', 'zip_clean', 'tract', 'block',
                    'treatment_intensity', 'high_block_intensity', 'cluster_size']
    numeric_data = data.select_dtypes(include=[np.number])
    all_features = [col for col in numeric_data.columns if col not in exclude_cols]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(data[all_features], data['voted'])

    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    selected_features = feature_importance[
        feature_importance['importance'] > feature_selection_threshold
        ]['feature'].tolist()

    block_data = data.groupby('block').agg({
        **{feat: 'mean' for feat in selected_features},
        'voted': 'mean'
    }).reset_index()

    block_to_idx = {block: idx for idx, block in enumerate(block_data['block'])}
    block_data['node_idx'] = block_data['block'].map(block_to_idx)

    block_features = block_data[['block'] + selected_features].copy()
    block_features['block_num'] = block_features['block'].astype('category').cat.codes

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')
    nbrs.fit(block_features[['block_num']].values)
    distances, indices = nbrs.kneighbors(block_features[['block_num']].values)

    edge_list = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_list.append([i, j])
            edge_list.append([j, i])

    if len(edge_list) == 0:
        raise ValueError("No edges created in graph")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    x = torch.tensor(block_data[selected_features].values, dtype=torch.float)
    y = torch.tensor(block_data['voted'].values, dtype=torch.float).unsqueeze(1)

    num_nodes = len(block_data)
    indices = np.arange(num_nodes)

    y_binary = (y.numpy() > 0.5).astype(int).flatten()
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_binary)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    model = GNNVotingModel(input_dim=len(selected_features), hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(graph_data.x, graph_data.edge_index)

        train_preds = predictions[graph_data.train_mask].numpy().flatten()
        train_labels = graph_data.y[graph_data.train_mask].numpy().flatten()

        test_preds = predictions[graph_data.test_mask].numpy().flatten()
        test_labels = graph_data.y[graph_data.test_mask].numpy().flatten()

        train_mse = mean_squared_error(train_labels, train_preds)
        train_mae = mean_absolute_error(train_labels, train_preds)
        train_r2 = r2_score(train_labels, train_preds)

        test_mse = mean_squared_error(test_labels, test_preds)
        test_mae = mean_absolute_error(test_labels, test_preds)
        test_r2 = r2_score(test_labels, test_preds)

    results = {
        'model': model,
        'graph_data': graph_data,
        'selected_features': selected_features,
        'feature_importance': feature_importance,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1] // 2,
        'num_features': len(selected_features)
    }

    return results

print(run_gnn_analysis())
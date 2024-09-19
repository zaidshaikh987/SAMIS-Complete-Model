import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Define a simple GNN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

# Function to create a graph structure
def create_graph_structure(df):
    G = nx.Graph()
    
    for idx, row in df.iterrows():
        node = row['region']
        G.add_node(node, price=row['Price'], temperature=row['Temperature'])
        # Add edges based on some logic (e.g., geographical proximity)
        # Here we just add random edges for demonstration
        for other_node in df['region'].unique():
            if node != other_node:
                G.add_edge(node, other_node)
    
    return G

# Function to convert graph to PyTorch Geometric Data
def graph_to_pyg_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor([list(G.nodes[node].values()) for node in G.nodes], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

# Training function
def train_gnn(data_loader, model, optimizer):
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
    print('Training complete.')

# Main function
def main():
    # Load data
    df = pd.read_csv('path/to/your/data.csv')
    
    # Create graph and convert to PyTorch Geometric Data
    G = create_graph_structure(df)
    pyg_data = graph_to_pyg_data(G)
    
    # Create a DataLoader
    data_loader = DataLoader([pyg_data], batch_size=1, shuffle=True)
    
    # Initialize model, optimizer
    model = GCN(in_channels=2, hidden_channels=16, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    train_gnn(data_loader, model, optimizer)

if __name__ == "__main__":
    main()

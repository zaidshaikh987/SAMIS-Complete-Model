import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Define a simple neural network for local models
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Federated Learning class
class FederatedLearning:
    def __init__(self, num_local_models, input_dim, output_dim, learning_rate=0.01):
        self.local_models = [SimpleNN(input_dim, output_dim) for _ in range(num_local_models)]
        self.global_model = SimpleNN(input_dim, output_dim)
        self.learning_rate = learning_rate

    def train_local_models(self, datasets, epochs=5):
        for i, model in enumerate(self.local_models):
            print(f"Training local model {i+1}/{len(self.local_models)}")
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            data_loader = DataLoader(datasets[i], batch_size=32, shuffle=True)
            
            for epoch in range(epochs):
                for inputs, targets in data_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
    
    def aggregate_updates(self):
        global_state_dict = self.global_model.state_dict()
        local_state_dicts = [model.state_dict() for model in self.local_models]
        
        # Average the state_dicts of the local models
        for key in global_state_dict:
            global_state_dict[key] = torch.mean(
                torch.stack([state_dict[key].float() for state_dict in local_state_dicts]), dim=0
            )
        
        self.global_model.load_state_dict(global_state_dict)

    def evaluate_global_model(self, test_data):
        self.global_model.eval()
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.global_model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        print(f"Global model evaluation loss: {avg_loss}")

def main():
    # Load datasets for local models
    df = pd.read_csv('path/to/your/data.csv')
    data = df.values
    num_local_models = 3
    input_dim = data.shape[1] - 1  # Assuming last column is target
    output_dim = 1  # Assuming single target column
    
    # Create local datasets
    datasets = [torch.utils.data.TensorDataset(torch.FloatTensor(data[:, :-1]), torch.FloatTensor(data[:, -1])) for _ in range(num_local_models)]
    
    # Initialize federated learning system
    federated_learning = FederatedLearning(num_local_models, input_dim, output_dim)
    
    # Train local models
    federated_learning.train_local_models(datasets)
    
    # Aggregate updates
    federated_learning.aggregate_updates()
    
    # Evaluate global model
    test_data = torch.utils.data.TensorDataset(torch.FloatTensor(data[:, :-1]), torch.FloatTensor(data[:, -1]))  # Assuming same data for testing
    federated_learning.evaluate_global_model(test_data)

if __name__ == "__main__":
    main()

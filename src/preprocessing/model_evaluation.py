import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Constants
MODEL_PATHS = {
    'prophet': 'path/to/prophet_model.pkl',
    'transformer': 'path/to/transformer_model.pkl',
    'gnn': 'path/to/gnn_model.pkl'
}
TEST_DATA_PATH = 'path/to/your/test_data.csv'

# Function to load test data
def load_test_data():
    test_data = pd.read_csv(TEST_DATA_PATH)
    return test_data

# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def main():
    # Load test data
    test_data = load_test_data()
    X_test = test_data.drop('target', axis=1)  # Assuming 'target' is the column to predict
    y_test = test_data['target']
    
    # Evaluate each model
    results = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            mse, mae, r2 = evaluate_model(model, X_test, y_test)
            results[model_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        else:
            results[model_name] = {'MSE': None, 'MAE': None, 'R2': None}
            print(f"Model {model_name} not found at {model_path}")
    
    # Print or save the results
    print("Model Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE={metrics['MSE']}, MAE={metrics['MAE']}, R2={metrics['R2']}")

if __name__ == "__main__":
    main()

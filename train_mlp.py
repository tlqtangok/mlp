import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import json

# The MLP class remains the same
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._layers = nn.Sequential(
            nn.Linear(self._input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, self._output_size)
        )

    def forward(self, x):
        return self._layers(x)

# Data loading function remains the same
def load_and_preprocess_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler

# Training function remains the same
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] finished.")
    print("Training finished.")

# New function to save the model and export to ONNX
def export_model_to_onnx(model, scaler, input_size):
    """
    Saves the PyTorch model, exports it to ONNX, and saves the scaler parameters.
    """
    # 1. Save the PyTorch model's state dictionary
    torch_model_path = "mlp_classifier.pth"
    torch.save(model.state_dict(), torch_model_path)
    print(f"PyTorch model saved to {torch_model_path}")

    # 2. Export the model to ONNX format
    onnx_model_path = "mlp_classifier.onnx"
    # Create a dummy input with the correct shape
    dummy_input = torch.randn(1, input_size, requires_grad=True)
    model.eval()

    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      onnx_model_path,     # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,    # the ONNX version to export the model to
                      do_constant_folding=True,
                      input_names=['input'],   # the model's input names
                      output_names=['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'}, # variable length axes
                                    'output' : {0 : 'batch_size'}})
    print(f"Model exported to ONNX format at {onnx_model_path}")

    # 3. Save the scaler parameters (mean and scale) for C++ inference
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    scaler_path = "scaler_params.json"
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)
    print(f"Scaler parameters saved to {scaler_path}")


# --- Main Execution ---
if __name__ == "__main__":
    num_samples = 1000
    num_features = 10
    num_classes = 3
    
    features = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, num_classes, num_samples)
    
    df_data = np.hstack([features, labels.reshape(-1, 1)])
    columns = [f'c{i}' for i in range(num_features)] + ['c10']
    data = pd.DataFrame(df_data, columns=columns)
    data['c10'] = data['c10'].astype(int)

    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(data)

    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_features = X_train.shape[1]
    mlp_model = MLP(input_size=input_features, output_size=num_classes)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

    train_model(mlp_model, train_loader, loss_function, optimizer, epochs=50)

    # --- Export the trained model ---
    export_model_to_onnx(mlp_model, scaler, input_features)




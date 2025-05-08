import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import datetime

# Import technical indicators module
# Note: Make sure technical_indicators.py is in the same directory or in your Python path
from technical_indicators import add_technical_indicators, generate_trading_signals

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration parameters
WINDOW_SIZE = 30  # Number of days to use for prediction
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 40
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
TRANSFORMER_HEADS = 8  # Number of attention heads in transformer
TRANSFORMER_LAYERS = 3  # Number of transformer layers
USE_TECHNICAL_INDICATORS = True  # Flag to enable/disable technical indicators
USE_TRADING_SIGNALS = True  # Flag to enable/disable trading signals

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# File paths
crypto_files = {
    'BTC': 'BitcoinData2025.csv',
    'ETH': 'EthereumData2025.csv',
    'DOGE': 'DogecoinData2025.csv'
}

# Define cutoff date
train_end_date = pd.to_datetime('2024-12-01')

# Custom dataset for sequence data
class CryptoDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Data preprocessing function with technical indicators
def preprocess_data(file_path, window_size):
    # Read data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (newest to oldest in the file)
    df = df.sort_values('Date')
    
    # Extract features
    base_features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Sentiment']
    
    # Clean volume data (remove commas, convert K and M)
    df['Vol.'] = df['Vol.'].astype(str).str.replace(',', '')
    df['Vol.'] = df['Vol.'].apply(lambda x: float(x[:-1]) * 1000 if 'K' in x else 
                                  float(x[:-1]) * 1000000 if 'M' in x else float(x))
    
    # Clean price columns (remove commas and convert to float)
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Extract the 'Change %' as a numeric feature
    df['Change %'] = df['Change %'].str.rstrip('%').astype(float) / 100.0
    
    # Rename columns to match technical indicators requirements
    df_for_indicators = df.rename(columns={
        'Price': 'Close',
        'Vol.': 'Volume'
    })
    
    # Add technical indicators if enabled
    if USE_TECHNICAL_INDICATORS:
        print("Adding technical indicators...")
        df_with_indicators = add_technical_indicators(df_for_indicators)
        
        # Add trading signals if enabled
        if USE_TRADING_SIGNALS:
            print("Generating trading signals...")
            df_with_indicators = generate_trading_signals(df_with_indicators)
            
            # Add signal columns to features
            signal_features = [col for col in df_with_indicators.columns if 'Flag' in col]
            print(f"Added {len(signal_features)} trading signal features")
        
        # Merge back with original dataframe
        # First, copy the technical indicators to the original dataframe
        technical_indicators = [col for col in df_with_indicators.columns 
                               if col not in df_for_indicators.columns]
        
        for col in technical_indicators:
            df[col] = df_with_indicators[col].values
        
        # Update features list to include technical indicators
        additional_features = [
            'MACD', 'MACD_Signal', 'MACD_Histogram', 
            'SMA20', 'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Width', 'Bollinger_Pct',
            'Stoch_K', 'Stoch_D', 
            'OBV', 
            'ATR', 
            'ROC', 
            'CCI', 
            'RSI'
        ]
        
        # Add Ichimoku features if they exist (they might be NaN at the beginning)
        ichimoku_features = [
            'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span'
        ]
        
        # Add MFI if it exists
        if 'MFI' in df.columns:
            additional_features.append('MFI')
            
        # Add trading signal features if enabled
        if USE_TRADING_SIGNALS:
            additional_features.extend(signal_features)
            
        # Filter out features that don't exist in the dataframe
        additional_features = [f for f in additional_features if f in df.columns]
        
        # Combine base features with additional features
        features = base_features + additional_features
        print(f"Using {len(features)} features: {features}")
    else:
        # Use only base features
        features = base_features
        print(f"Using {len(features)} base features: {features}")
    
    # Extract feature data
    feature_data = df[features].values
    dates = df['Date'].values
    
    # Split the data
    train_mask = df['Date'] <= train_end_date
    test_mask = df['Date'] > train_end_date
    
    # Get training and testing data
    train_data = feature_data[train_mask]
    test_data = feature_data[test_mask]
    train_dates = dates[train_mask]
    test_dates = dates[test_mask]
    
    # Handle NaN values (from technical indicators that need historical data)
    # Replace NaNs with 0 for simplicity
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    def create_sequences(data, window_size):
        xs, ys = [], []
        for i in range(len(data) - window_size):
            x = data[i:i+window_size]
            y = data[i+window_size, 0]  # Price/Close is the target (first column)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    # Create training sequences
    X_train, y_train = create_sequences(train_scaled, window_size)
    
    # For testing, we need to include the last window_size elements from training
    # to predict the first test point
    combined_data = np.vstack([train_scaled[-window_size:], test_scaled])
    X_test, y_test = create_sequences(combined_data, window_size)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Get actual prices for evaluation
    test_actual_prices = df.loc[test_mask, 'Price'].values
    
    return (
        X_train_tensor, y_train_tensor, 
        X_test_tensor, y_test_tensor,
        train_dates[window_size:], test_dates,  
        scaler, test_actual_prices
    )

# Create data loaders
def create_data_loaders(X_train, y_train, X_test, y_test, batch_size):
    train_dataset = CryptoDataset(X_train, y_train)
    test_dataset = CryptoDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Model definitions
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.0):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # RNN output: (batch_size, seq_len, hidden_size)
        output, _ = self.rnn(x, h0)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(output * attention_weights, dim=1)
        
        # Final prediction
        return self.fc(context_vector)

class ComplexRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.2):
        super(ComplexRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Multiple fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # RNN output
        output, _ = self.rnn(x, h0)
        
        # Apply attention
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(output * attn_weights, dim=1)
        
        # Fully connected layers
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU output
        output, _ = self.gru(x, h0)
        
        # Apply attention
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(output * attn_weights, dim=1)
        
        # Final prediction
        return self.fc(context)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM output
        output, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(output * attn_weights, dim=1)
        
        # Final prediction
        return self.fc(context)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, nhead=8, num_layers=3, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.input_linear = nn.Linear(input_size, hidden_size)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, src):
        # Transform input to match transformer dimensions
        src = self.input_linear(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer
        output = self.transformer_encoder(src)
        
        # Use the mean of all sequence positions for the final prediction
        output = output.mean(dim=1)
        
        return self.decoder(output)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs, model_name, crypto_name):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], {crypto_name} {model_name} Loss: {avg_loss:.6f}')
    
    return losses

# Testing function
def test_model(model, test_loader, scaler, actual_prices):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
    
    # Create a dummy array to inverse transform
    dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy[:, 0] = predictions
    
    # Inverse transform to get the actual price predictions
    predictions_rescaled = scaler.inverse_transform(dummy)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(actual_prices, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predictions_rescaled)
    r2 = r2_score(actual_prices, predictions_rescaled)
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    return predictions_rescaled, mse, rmse, mae, r2

# Function to save models
def save_model(model, file_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Function to visualize predictions
def plot_predictions(dates, actual_prices, predictions_dict, title, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # Plot actual prices
    plt.plot(dates, actual_prices, label='Actual Price', color='black', linewidth=2)
    
    # Plot predictions for each model
    for model_name, preds in predictions_dict.items():
        plt.plot(dates, preds, label=f'{model_name} Prediction', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# Function to plot training losses
def plot_losses(losses_dict, title, save_path=None):
    plt.figure(figsize=(12, 6))
    
    for model_name, losses in losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f'{model_name}')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# Function to plot feature importance
def plot_feature_importance(model, feature_names, title, save_path=None):
    # Only works for models that have feature_importances_ attribute (like Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    else:
        print("Model does not support feature importance visualization")

# Function to run entire pipeline for a cryptocurrency
def run_crypto_pipeline(crypto_name, file_path):
    print(f"\n{'='*50}")
    print(f"Processing {crypto_name}")
    print(f"{'='*50}")
    
    # Preprocess data
    X_train, y_train, X_test, y_test, train_dates, test_dates, scaler, actual_prices = preprocess_data(file_path, WINDOW_SIZE)
    
    # Get input size from data
    input_size = X_train.shape[2]
    print(f"Input size (number of features): {input_size}")
    
    # Create dataloaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, BATCH_SIZE)
    
    # Initialize models
    models = {
        'SimpleRNN': SimpleRNN(input_size, HIDDEN_SIZE, num_layers=1).to(device),
        'ComplexRNN': ComplexRNN(input_size, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device),
        'GRU': GRUModel(input_size, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device),
        'LSTM': LSTMModel(input_size, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device),
        'Transformer': TransformerModel(input_size, HIDDEN_SIZE, nhead=TRANSFORMER_HEADS, num_layers=TRANSFORMER_LAYERS, dropout=DROPOUT).to(device)
    }
    
    # Training and testing
    losses_dict = {}
    predictions_dict = {}
    metrics_dict = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} for {crypto_name}...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train the model
        losses = train_model(model, train_loader, criterion, optimizer, EPOCHS, model_name, crypto_name)
        losses_dict[model_name] = losses
        
        # Test the model
        print(f"\nTesting {model_name} for {crypto_name}...")
        predictions, mse, rmse, mae, r2 = test_model(model, test_loader, scaler, actual_prices)
        predictions_dict[model_name] = predictions
        metrics_dict[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        # Save the model
        model_dir = f"models/{crypto_name}/"
        os.makedirs(model_dir, exist_ok=True)
        save_model(model, f"{model_dir}{model_name}_with_indicators.h5")
    
    # Visualize results
    plot_dir = "plots/"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot predictions
    plot_predictions(
        test_dates,
        actual_prices,
        predictions_dict,
        f"{crypto_name} Price Predictions (With Technical Indicators)",
        f"{plot_dir}{crypto_name}_predictions_with_indicators.png"
    )
    
    # Plot losses
    plot_losses(
        losses_dict,
        f"{crypto_name} Training Loss (With Technical Indicators)",
        f"{plot_dir}{crypto_name}_losses_with_indicators.png"
    )
    
    return metrics_dict

# Main execution
def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    all_metrics = {}
    
    # Process each cryptocurrency
    for crypto_name, file_path in crypto_files.items():
        metrics = run_crypto_pipeline(crypto_name, file_path)
        all_metrics[crypto_name] = metrics
    
    # Print final metrics
    print("\n\n===== FINAL RESULTS (WITH TECHNICAL INDICATORS) =====")
    for crypto_name, models_metrics in all_metrics.items():
        print(f"\n{crypto_name} Results:")
        print("-" * 50)
        for model_name, metrics in models_metrics.items():
            print(f"{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            print("-" * 30)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 7 20:48:10 2025

@author: A K @ getgh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class StockDataProcessor:
    """Class to handle data download, preprocessing, and feature engineering"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        
    def download_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Downloading data for {self.symbol}...")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} trading days of data")
        return self.data
    
    def engineer_features(self):
        """Create momentum and technical indicators"""
        df = self.data.copy()
        
        # Basic price features
        df['Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
        
        # Momentum indicators
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Lower'], df['BB_Middle'] = self.calculate_bollinger_bands(df['Close'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Past returns (momentum signals)
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
            df[f'Log_Return_Lag_{lag}'] = df['Log_Return'].shift(lag)
        
        # Rolling statistics
        df['Return_Vol_5'] = df['Return'].rolling(window=5).std()
        df['Return_Vol_20'] = df['Return'].rolling(window=20).std()
        df['Return_Mean_5'] = df['Return'].rolling(window=5).mean()
        df['Return_Mean_20'] = df['Return'].rolling(window=20).mean()
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # Target variable (next day return)
        df['Target'] = df['Return'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        self.features = df
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean

class StockDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class StockPredictor:
    
    def __init__(self, sequence_length=30, test_size=0.2, val_size=0.1):
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, features_df):
        """Prepare data for training"""
        # Select feature columns (exclude target and non-numeric columns)
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Handle any remaining NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        X = features_df[feature_cols].values
        y = features_df['Target'].values
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Split data
        total_samples = len(X)
        train_size = int(total_samples * (1 - self.test_size - self.val_size))
        val_size = int(total_samples * self.val_size)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
        return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq)
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """Build LSTM model"""
        self.model = LSTMModel(input_size, hidden_size, num_layers, 1, dropout)
        return self.model
    
    def train_model(self, train_data, val_data, epochs=100, batch_size=64, learning_rate=0.001):
        """Train the LSTM model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create data loaders
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred.squeeze(), y_batch)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
    
    def predict(self, X):
        """predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            return predictions.numpy()
    
    def evaluate_model(self, test_data):
        """E-val model performance"""
        X_test, y_test = test_data
        
        # Make predictions
        y_pred_scaled = self.predict(X_test)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        direction_accuracy = np.mean(np.sign(y_pred.flatten()) == np.sign(y_true.flatten()))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Direction Accuracy': direction_accuracy
        }
        
        return metrics, y_pred.flatten(), y_true.flatten()

class PolynomialBenchmark:
    """ Polynomial regression benchmark model"""
    
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_data(self, X, y):
        """Prepare data for polynomial regression"""
        # Use last sequence values as features
        X_last = X[:, -1, :]  # Take last time step
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X_last)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create polynomial features
        X_poly = self.poly_features.fit_transform(X_scaled)
        
        return X_poly, y_scaled
    
    def train(self, X, y):
        """ Train polynomial regression model"""
        X_poly, y_scaled = self.prepare_data(X, y)
        self.model.fit(X_poly, y_scaled)
        
    def predict(self, X, y):
        """Make predictions"""
        X_poly, _ = self.prepare_data(X, y)
        y_pred_scaled = self.model.predict(X_poly)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred.flatten()

class TradingStrategy:
    """Simple long-short trading strategy"""
    
    def __init__(self, threshold=0.001):
        self.threshold = threshold
        
    def generate_signals(self, predictions):
        """Generate trading signals based on predictions"""
        signals = np.zeros(len(predictions))
        signals[predictions > self.threshold] = 1   # Long
        signals[predictions < -self.threshold] = -1  # Short
        return signals
    
    def calculate_returns(self, signals, actual_returns):
        """Calculate strategy returns"""
        strategy_returns = signals * actual_returns
        return strategy_returns
    
    def calculate_metrics(self, strategy_returns, actual_returns):
        """Calculate strategy performance metrics"""
        cumulative_returns = np.cumsum(strategy_returns)
        cumulative_market = np.cumsum(actual_returns)
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        
        # Maximum drawdown
        cumulative_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(strategy_returns > 0)
        
        # Information ratio
        excess_returns = strategy_returns - actual_returns
        info_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        metrics = {
            'Total Return': cumulative_returns[-1],
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Information Ratio': info_ratio
        }
        
        return metrics, cumulative_returns, cumulative_market

# Main execution
def main():
    # Parameters
    SYMBOL = "AAPL"  # Change to any S&P 500 stock
    START_DATE = "2010-01-01"
    END_DATE = "2024-01-01"
    
    print("=== Stock Momentum Forecasting with LSTM ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    
    # 1. Data Processing
    processor = StockDataProcessor(SYMBOL, START_DATE, END_DATE)
    raw_data = processor.download_data()
    features_df = processor.engineer_features()
    
    print(f"\nFeatures created: {len(features_df.columns)} columns")
    print(f"Data shape: {features_df.shape}")
    
    # 2. Prepare data for LSTM
    predictor = StockPredictor(sequence_length=30, test_size=0.2, val_size=0.1)
    train_data, val_data, test_data = predictor.prepare_data(features_df)
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    print(f"\nTrain sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    input_size = X_train.shape[2]
    lstm_model = predictor.build_model(input_size, hidden_size=64, num_layers=2)
    
    print(f"\nTraining LSTM model...")
    train_losses, val_losses = predictor.train_model(
        train_data, val_data, epochs=50, batch_size=32
    )
    
    # 4. Evaluate LSTM model
    lstm_metrics, lstm_pred, y_true = predictor.evaluate_model(test_data)
    
    print(f"\n=== LSTM Model Performance ===")
    for metric, value in lstm_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 5. Polynomial regression benchmark
    print(f"\nTraining Polynomial Regression benchmark...")
    poly_model = PolynomialBenchmark(degree=2)
    poly_model.train(X_train, y_train)
    poly_pred = poly_model.predict(X_test, y_test)
    
    # Evaluate polynomial model
    poly_mse = mean_squared_error(y_true, poly_pred)
    poly_direction_acc = np.mean(np.sign(poly_pred) == np.sign(y_true))
    
    print(f"\n=== Polynomial Regression Performance ===")
    print(f"MSE: {poly_mse:.6f}")
    print(f"Direction Accuracy: {poly_direction_acc:.4f}")
    
    # 6. Trading strategy evaluation
    print(f"\n=== Trading Strategy Analysis ===")
    strategy = TradingStrategy(threshold=0.001)
    
    # LSTM strategy
    lstm_signals = strategy.generate_signals(lstm_pred)
    lstm_strategy_returns = strategy.calculate_returns(lstm_signals, y_true)
    lstm_strategy_metrics, lstm_cum_returns, market_cum_returns = strategy.calculate_metrics(
        lstm_strategy_returns, y_true
    )
    
    print(f"\nLSTM Strategy:")
    for metric, value in lstm_strategy_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Polynomial strategy
    poly_signals = strategy.generate_signals(poly_pred)
    poly_strategy_returns = strategy.calculate_returns(poly_signals, y_true)
    poly_strategy_metrics, poly_cum_returns, _ = strategy.calculate_metrics(
        poly_strategy_returns, y_true
    )
    
    print(f"\nPolynomial Strategy:")
    for metric, value in poly_strategy_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 7. Visualization
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Training losses
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot 2: Predictions vs Actual
    plt.subplot(2, 3, 2)
    plt.scatter(y_true, lstm_pred, alpha=0.5, label='LSTM')
    plt.scatter(y_true, poly_pred, alpha=0.5, label='Polynomial')
    plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Predictions vs Actual Returns')
    plt.legend()
    
    # Plot 3: Cumulative returns
    plt.subplot(2, 3, 3)
    plt.plot(lstm_cum_returns, label='LSTM Strategy')
    plt.plot(poly_cum_returns, label='Polynomial Strategy')
    plt.plot(market_cum_returns, label='Buy & Hold')
    plt.title('Cumulative Returns')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.hist(lstm_strategy_returns, bins=50, alpha=0.7, label='LSTM Strategy')
    plt.hist(poly_strategy_returns, bins=50, alpha=0.7, label='Polynomial Strategy')
    plt.hist(y_true, bins=50, alpha=0.7, label='Market Returns')
    plt.title('Return Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    feature_corr = features_df[predictor.feature_columns + ['Target']].corr()['Target'].abs().sort_values(ascending=False)[1:11]
    plt.barh(range(len(feature_corr)), feature_corr.values)
    plt.yticks(range(len(feature_corr)), feature_corr.index)
    plt.title('Top 10 Feature Correlations with Target')
    plt.xlabel('Absolute Correlation')
    
    # Plot 6: Signal distribution
    plt.subplot(2, 3, 6)
    signal_counts = pd.Series(lstm_signals).value_counts()
    plt.pie(signal_counts.values, labels=['Hold', 'Short', 'Long'], autopct='%1.1f%%')
    plt.title('LSTM Signal Distribution')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Model Comparison Summary ===")
    print(f"LSTM RMSE: {lstm_metrics['RMSE']:.6f}")
    print(f"Polynomial RMSE: {np.sqrt(poly_mse):.6f}")
    print(f"LSTM Direction Accuracy: {lstm_metrics['Direction Accuracy']:.4f}")
    print(f"Polynomial Direction Accuracy: {poly_direction_acc:.4f}")
    print(f"LSTM Strategy Sharpe: {lstm_strategy_metrics['Sharpe Ratio']:.4f}")
    print(f"Polynomial Strategy Sharpe: {poly_strategy_metrics['Sharpe Ratio']:.4f}")

if __name__ == "__main__":
    main()

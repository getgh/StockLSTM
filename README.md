Stock momentum with LSTM
Hey there! 👋 This is a project I built to predict stock prices using machine learning. Specifically, I'm using LSTM neural networks to forecast whether a stock will go up or down tomorrow based on technical indicators and momentum signals.
What does this thing do?
Basically, I feed the model 10+ years of stock data along with technical indicators (like RSI, MACD, moving averages) and it tries to predict the next day's return. Then I test it in a simple trading strategy to see if it can actually make money.
Think of it like teaching a computer to be a day trader, but with math instead of gut feelings.
Why I built this
I was curious about whether machine learning could actually beat traditional methods for stock prediction. Plus, I wanted to learn more about:

Time series forecasting with neural networks*
Technical analysis indicators!
How to properly backtest trading strategies?
PyTorch for financial applications.

Data Pipeline: Ues to DL stock data from yFinance and creates features
Feature Engineering: Calculates 25+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
LSTM Model: A neural network that learns from sequences of past data
Benchmark: Compares against polynomial regression (spoiler: LSTM usually wins)
Trading Strategy: Tests predictions with a simple long/short strategy
Visualizations: Pretty charts showing how well (or poorly) it performs

To run

Please use py 3.8 or newer
A few Python libraries

Installation:
bashgit clone https://github.com/getgh/StockLSTM
cd stock-momentum-lstm
pip install -r requirements.txt

bashpython stock_momentum_lstm.py
After > It'll download Apple (AAPL) data by default and start training.
To try different stocks
Changing this line of code will work:
pythonSYMBOL = "MSFT"  # Try Microsoft, Tesla (TSLA), Google (GOOGL), etc.

It will:
Download ~10 years of stock data
Create some of technical indicators
Train an LSTM model (may take a while)
Test it against polynomial regression
Show you pretty charts and performance metrics
LEt us know if the strategy would have made money


Direction Accuracy: ~55-60% (slightly better than random)
Sharpe Ratio: Varies a lot, sometimes beats buy-and-hold
LSTM vs Polynomial: LSTM usually wins, but not by huge margins

The model tends to work better during trending markets and struggles during choppy sideways action (just like human traders).
Files you'll find
stock-momentum-lstm/
├── main.py                     # main
├── requirements.txt            # What to install
├── README.md                   # (Currently reading)
└── results/                    # Charts and outputs

How it works (the nerdy stuff)
The Data

Downloads historical prices from Yahoo Finance
Calculates technical indicators like RSI, MACD, moving averages
Creates momentum features from past returns
Splits data properly (no peeking into the future!)

The Model

Uses LSTM (Long Short-Term Memory) neural networks
Looks at 30 days of history to predict tomorrow
Has dropout to prevent overfitting
Trained with early stopping to avoid memorizing noise

The Strategy

If predicted return > 0.1%, go long
If predicted return < -0.1%, go short
Otherwise, stay in cash
Calculates realistic performance metrics

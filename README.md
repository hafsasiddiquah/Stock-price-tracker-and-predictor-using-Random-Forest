# Stock Price Tracker & Predictor  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-orange)  
![Yahoo Finance](https://img.shields.io/badge/Data-YFinance-red)  

A machine learning-powered stock price tracker and predictor built with Random Forest Regressor, featuring technical indicators (RSI, MACD) and an interactive Streamlit dashboard.  

📌 Features  

Real-time stock data from Yahoo Finance  
Technical indicators (RSI, MACD, moving averages)  
Random Forest model with hyperparameter tuning (GridSearchCV)  
Next-day price prediction with confidence metrics  
Interactive visualizations (historical trends, predictions vs. actual)  
Feature importance analysis  

🚀 Quick Start  

1. Install dependencies:  
   ```bash
   pip install streamlit yfinance pandas numpy scikit-learn matplotlib
   ```

2. Run the app:  
   ```bash
   streamlit run fin.py
   ```

3. Enter a stock ticker (e.g., `AAPL`, `TSLA`) and adjust date ranges.  

## 📊 Example Output  

![image](https://github.com/user-attachments/assets/bcb204f9-2a29-43ae-8b7b-e307b49c1e66)
![image](https://github.com/user-attachments/assets/182cec20-1b5e-41e9-9581-8535ac742c3c)
![image](https://github.com/user-attachments/assets/841022d0-0697-475b-b23c-c480a8c3b050)
![image](https://github.com/user-attachments/assets/0122f74e-2b29-43f4-88b8-75e15ec14fae)

## 📈 Model Performance  
- Mean Absolute Error (MAE): 1.2–2.8%  

## 📝 Note  
Predictions are for educational purposes only. Not financial advice.  

Made for finance and ML enthusiasts. Contributions welcome! 🛠️

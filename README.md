# Thailand Macroeconomic Forecasting Engine

This project is an interactive web dashboard for forecasting key macroeconomic indicators for Thailand. It is built entirely in Python using Streamlit, pulling real data from FRED (OECD) and Yahoo Finance, and applying Econometric & Machine Learning models.

## Features
*   **Keyless Data APIs:** Uses `pandas-datareader` (FRED) and `yfinance` (Yahoo) to bypass API key requirements.
*   **Predictive Models:** Vector Autoregression (VAR), Meta Prophet, and Random Forest Classifier.
*   **Interactive Dashboard:** Built with Streamlit and Plotly for high-quality data visualization.

## Setup Instructions

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. A web browser will automatically open showing your dashboard.

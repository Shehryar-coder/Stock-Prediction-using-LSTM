## 🚀 Installation & Setup Guide

This project uses an **LSTM (Long Short-Term Memory)** model to predict stock prices. Because deep learning libraries like **TensorFlow** have specific version requirements, please follow these steps exactly.

### 📋 Prerequisites
* **Python 3.12.x** (Required for TensorFlow/Keras stability)
* **VS Code** (Recommended)

> **Note:** If you are using Python 3.14 or newer, TensorFlow will not install correctly. Please ensure you use the Python 3.12 interpreter.

### 🛠️ Step-by-Step Execution

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/your-username/StockPredictionLSTM.git](https://github.com/your-username/StockPredictionLSTM.git)
   cd StockPredictionLSTM

   Create a Virtual Environment
Using a virtual environment prevents library conflicts with your global Python installation.

PowerShell
# Create environment for Python 3.12
py -3.12 -m venv venv
Activate the Environment

PowerShell
# Windows (PowerShell)
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
You should now see (venv) at the start of your terminal line.

Install Dependencies
Install the full machine learning stack in one command:

PowerShell
pip install streamlit pandas numpy yfinance scikit-learn tensorflow keras
Run the Dashboard
Streamlit apps must be launched using the streamlit run command:

PowerShell
streamlit run dashboard.py

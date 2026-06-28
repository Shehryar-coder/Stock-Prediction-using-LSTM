# 🚀 Installation & Setup Guide

This project uses an **LSTM (Long Short-Term Memory)** model to predict stock prices. Since deep learning libraries such as **TensorFlow** have specific version requirements, please follow the instructions below carefully.

---

## 📋 Prerequisites

Before running the project, make sure you have:

* **Python 3.12.x** (Required for TensorFlow/Keras compatibility)
* **VS Code** (Recommended)

> **Important:** If you are using **Python 3.14 or newer**, TensorFlow may not install correctly. Please use a **Python 3.12 interpreter**.

---

## 🛠️ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/StockPredictionLSTM.git
cd StockPredictionLSTM
```

---

## 🛠️ Step 2: Create a Virtual Environment

Using a virtual environment helps prevent dependency conflicts.

### Windows (PowerShell)

```powershell
py -3.12 -m venv venv
```

### macOS / Linux

```bash
python3.12 -m venv venv
```

---

## 🛠️ Step 3: Activate the Virtual Environment

### Windows (PowerShell)

```powershell
.\venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

After activation, you should see:

```bash
(venv)
```

at the beginning of your terminal prompt.

---

## 🛠️ Step 4: Install Dependencies

Install all required libraries using:

```bash
pip install streamlit pandas numpy yfinance scikit-learn tensorflow keras
```

---

## 🛠️ Step 5: Run the Dashboard

Launch the Streamlit application using:

```bash
streamlit run dashboard.py
```

---

## ✅ Expected Result

After running the command above, Streamlit will start a local server and automatically open the Stock Prediction Dashboard in your web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 📦 Project Dependencies

* Streamlit
* Pandas
* NumPy
* Yahoo Finance (yfinance)
* Scikit-learn
* TensorFlow
* Keras

---

## ⚠️ Troubleshooting

### TensorFlow Installation Issues

Verify your Python version:

```bash
python --version
```

Make sure it returns:

```text
Python 3.12.x
```

If you are using Python 3.14 or newer, create a new virtual environment with Python 3.12 and reinstall the dependencies.


# Product Demand Forecasting System

## 📌 Project Overview

A machine learning-based web application for forecasting product demand using historical sales data.

The system uses historical sales patterns, seasonal information, and product-related features to generate demand predictions. The prediction functionality is integrated into a Django web application.

## 🛠️ Technologies Used

- Python
- Django
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- HTML
- CSS
- JavaScript
- SQLite

## 🤖 Machine Learning Models

The project includes different deep learning approaches for demand forecasting:

- Artificial Neural Network (ANN)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The project also includes preprocessing components such as:

- Feature scaling
- Label encoding
- Feature column configuration
- Seasonal information
- Historical demand data

## 📂 Project Structure

```text
Product-Demand-Forecasting/
│
└── sales_prediction/
    ├── predictor/
    │   ├── model/
    │   ├── static/
    │   ├── templates/
    │   ├── migrations/
    │   ├── models.py
    │   ├── views.py
    │   └── urls.py
    │
    ├── sales_prediction/
    ├── manage.py
    └── db.sqlite3

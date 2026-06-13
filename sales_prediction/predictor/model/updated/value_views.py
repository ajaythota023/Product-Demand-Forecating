import os
import numpy as np
import pandas as pd
import joblib

from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model


# =============================
# PATHS
# =============================

MODEL_DIR = os.path.join(settings.BASE_DIR, 'predictor', 'model', 'updated')

model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')

scaler_X_path = os.path.join(MODEL_DIR, 'scaler_X (1).pkl')
scaler_y_path = os.path.join(MODEL_DIR, 'scaler_y (1).pkl')

le_category_path = os.path.join(MODEL_DIR, 'le_category (1).pkl')
le_product_path = os.path.join(MODEL_DIR, 'le_product (1).pkl')
le_season_path = os.path.join(MODEL_DIR, 'le_season (1).pkl')

feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns (1).pkl')
seasonal_mean_map_path = os.path.join(MODEL_DIR, 'seasonal_mean_map.pkl')
base_year_path = os.path.join(MODEL_DIR, 'base_year.pkl')

dataset_path = os.path.join(MODEL_DIR, 'demand_forecasting_dataset.csv')


# =============================
# LOAD FILES
# =============================

model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='mse')

scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

le_category = joblib.load(le_category_path)
le_product = joblib.load(le_product_path)
le_season = joblib.load(le_season_path)

feature_columns = joblib.load(feature_columns_path)
seasonal_mean_map = joblib.load(seasonal_mean_map_path)
base_year = joblib.load(base_year_path)


# =============================
# LOAD DATASET (weekly → monthly)
# =============================

df = pd.read_csv(dataset_path)
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month

df = df.groupby(
    ['category','product_name','Year','Month','season','festival_flag']
)['units_sold'].sum().reset_index()

df.rename(columns={
    'category':'Category',
    'product_name':'Product Name',
    'season':'Season',
    'festival_flag':'Festival_Flag',
    'units_sold':'Sales'
}, inplace=True)

df.sort_values(["Product Name","Year","Month"], inplace=True)


# =============================
# SEASON DETECTION
# =============================

def get_season(month):
    if month in [3,4,5,6]:
        return "Summer"
    elif month in [7,8,9]:
        return "Monsoon"
    elif month in [12,1,2]:
        return "Winter"
    else:
        return "Autumn"


# =============================
# FEATURE PREPARATION
# =============================

def prepare_input(category, product, year, month, festival_flag, sales_series):

    category_enc = le_category.transform([category])[0]
    product_enc = le_product.transform([product])[0]
    season = get_season(month)
    season_enc = le_season.transform([season])[0]

    year_index = year - base_year

    lag_1 = sales_series[-1] if len(sales_series)>=1 else 500
    lag_2 = sales_series[-2] if len(sales_series)>=2 else lag_1
    lag_3 = sales_series[-3] if len(sales_series)>=3 else lag_1
    lag_12 = sales_series[-12] if len(sales_series)>=12 else np.mean(sales_series)
    # lag_24 = sales_series[-24] if len(sales_series)>=24 else np.mean(sales_series)

    rolling_3 = np.mean(sales_series[-3:]) if len(sales_series)>=3 else np.mean(sales_series)

    # seasonal_mean from training map
    seasonal_key = (product, month)
    seasonal_mean = seasonal_mean_map.get(seasonal_key, np.mean(sales_series))

    input_data = [[
        category_enc,
        product_enc,
        season_enc,
        festival_flag,
        month,
        year_index,
        lag_1,
        lag_2,
        lag_3,
        lag_12,
        rolling_3,
        seasonal_mean
    ]]

    return np.array(input_data)


# =============================
# MAIN VIEW
# =============================

def predict_sales(request):

    prediction = None
    demand_level = None
    error_message = None

    # 🔹 Preserve selected values
    selected_category = request.GET.get("category") or request.POST.get("category")
    selected_product = None
    selected_year = None
    selected_month = None

    categories = df["Category"].unique()

    category_products = {
        cat: df[df["Category"] == cat]["Product Name"].unique().tolist()
        for cat in categories
    }

    if request.method == "POST":
        try:
            category = request.POST.get("category")
            product = request.POST.get("product")
            target_year = int(request.POST.get("year"))
            target_month = int(request.POST.get("month"))

            # 🔹 Preserve values for template
            selected_product = product
            selected_year = target_year
            selected_month = target_month

            product_data = df[df["Product Name"] == product].copy()
            product_data.sort_values(["Year", "Month"], inplace=True)

            sales_series = product_data["Sales"].tolist()

            current_year = product_data["Year"].max()
            current_month = product_data[product_data["Year"] == current_year]["Month"].max()

            while (current_year < target_year) or \
                  (current_year == target_year and current_month < target_month):

                if current_month == 12:
                    current_month = 1
                    current_year += 1
                else:
                    current_month += 1

                festival_flag = 1 if current_month in [10, 11, 12, 1] else 0

                input_data = prepare_input(
                    category, product, current_year, current_month,
                    festival_flag, sales_series
                )

                input_scaled = scaler_X.transform(input_data)
                input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))

                pred_scaled = model.predict(input_scaled, verbose=0)
                next_sale = scaler_y.inverse_transform(pred_scaled)[0][0]

                next_sale = max(next_sale, 0)
                sales_series.append(next_sale)

            prediction = float(sales_series[-1])

            # 🔹 Demand classification
            min_sales = product_data["Sales"].min()
            max_sales = product_data["Sales"].max()

            low_threshold = min_sales + (max_sales - min_sales) * 0.3
            high_threshold = min_sales + (max_sales - min_sales) * 0.7

            if prediction >= high_threshold:
                demand_level = "High"
            elif prediction >= low_threshold:
                demand_level = "Medium"
            else:
                demand_level = "Low"

        except Exception as e:
            error_message = str(e)

    context = {
        "prediction": prediction,
        "demand_level": demand_level,
        "error_message": error_message,
        "categories": categories,
        "category_products": category_products,
        "selected_category": selected_category,
        "selected_product": selected_product,
        "selected_year": selected_year,
        "selected_month": selected_month,
        "months": [
            {"num": i, "name": pd.Timestamp(2024, i, 1).strftime('%B')}
            for i in range(1, 13)
        ]
    }

    return render(request, "predict.html", context)
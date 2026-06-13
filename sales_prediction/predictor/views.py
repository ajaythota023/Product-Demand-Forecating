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

MODEL_DIR = os.path.join(settings.BASE_DIR, 'predictor', 'model', 'final')

model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')

scaler_X_path = os.path.join(MODEL_DIR, 'scaler_X.pkl')
scaler_y_path = os.path.join(MODEL_DIR, 'scaler_y.pkl')

le_category_path = os.path.join(MODEL_DIR, 'le_category.pkl')
le_product_path = os.path.join(MODEL_DIR, 'le_product.pkl')
le_season_path = os.path.join(MODEL_DIR, 'le_season.pkl')

feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.pkl')
seasonal_mean_map_path = os.path.join(MODEL_DIR, 'seasonal_mean_map.pkl')
base_year_path = os.path.join(MODEL_DIR, 'base_year.pkl')

dataset_path = os.path.join(MODEL_DIR, 'final_combined_demand_dataset.csv')


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
    ['category', 'product_name', 'Year', 'Month', 'season', 'festival_flag']
)['units_sold'].sum().reset_index()

df.rename(columns={
    'category': 'Category',
    'product_name': 'Product Name',
    'season': 'Season',
    'festival_flag': 'Festival_Flag',
    'units_sold': 'Sales'
}, inplace=True)

df.sort_values(["Product Name", "Year", "Month"], inplace=True)


# =============================
# SEASON DETECTION
# =============================

def get_season(month):
    if month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    elif month in [12, 1, 2]:
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

    lag_1 = sales_series[-1] if len(sales_series) >= 1 else 500
    lag_2 = sales_series[-2] if len(sales_series) >= 2 else lag_1
    lag_3 = sales_series[-3] if len(sales_series) >= 3 else lag_1
    lag_12 = sales_series[-12] if len(sales_series) >= 12 else np.mean(sales_series)

    rolling_3 = np.mean(sales_series[-3:]) if len(sales_series) >= 3 else np.mean(sales_series)

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
# HELPER: Forecast a single next step
# =============================

def forecast_one_step(category, product, year, month, sales_series):
    festival_flag = 1 if month in [10, 11, 12, 1] else 0
    input_data = prepare_input(
        category, product, year, month, festival_flag, sales_series
    )
    input_scaled = scaler_X.transform(input_data)
    input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    pred_scaled = model.predict(input_scaled, verbose=0)
    next_sale = scaler_y.inverse_transform(pred_scaled)[0][0]
    return max(next_sale, 0)


# =============================
# HELPER: Forecast all 12 months of a year
# =============================

def forecast_full_year(category, product, target_year, sales_series_snapshot):
    sales_series = list(sales_series_snapshot)
    monthly_predictions = {}

    for month in range(1, 13):
        predicted = forecast_one_step(
            category, product, target_year, month, sales_series
        )
        monthly_predictions[month] = predicted
        sales_series.append(predicted)

    return monthly_predictions


# =============================
# HELPER: Advance sales series to end of (target_year - 1)
# =============================

def advance_series_to_base(category, product, target_year, product_data, sales_series):
    current_year = product_data["Year"].max()
    current_month = product_data[
        product_data["Year"] == current_year
    ]["Month"].max()

    temp_year = current_year
    temp_month = current_month
    temp_series = list(sales_series)

    while temp_year < target_year - 1 or (
        temp_year == target_year - 1 and temp_month < 12
    ):
        if temp_month == 12:
            temp_month = 1
            temp_year += 1
        else:
            temp_month += 1

        predicted = forecast_one_step(
            category, product, temp_year, temp_month, temp_series
        )
        temp_series.append(predicted)

    return temp_series


# =============================
# HELPER: Compute reference value
# =============================

def get_reference_value(product_data):
    product_data = product_data.copy()
    product_data["year_index"] = product_data["Year"] - base_year
    product_data["trend_multiplier"] = 1 + (0.04 * product_data["year_index"])
    product_data["detrended_sales"] = (
        product_data["Sales"] / product_data["trend_multiplier"]
    )
    return product_data["detrended_sales"].tail(12).max()


# =============================
# HELPER: Classify demand level
# =============================

def classify_demand(demand_percentage):
    if demand_percentage >= 70:
        return "High"
    elif demand_percentage >= 40:
        return "Medium"
    else:
        return "Low"


# =============================
# HOME VIEW
# =============================

def home(request):
    return render(request, "home.html")


# =============================
# PREDICT SALES VIEW
# =============================

def predict_sales(request):

    prediction = None
    demand_level = None
    demand_percentage = None
    error_message = None

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

            selected_product = product
            selected_year = target_year
            selected_month = target_month

            product_data = df[df["Product Name"] == product].copy()
            product_data.sort_values(["Year", "Month"], inplace=True)

            sales_series = product_data["Sales"].tolist()

            current_year = product_data["Year"].max()
            current_month = product_data[
                product_data["Year"] == current_year
            ]["Month"].max()

            # Recursive forecast to target month only
            while (current_year < target_year) or \
                  (current_year == target_year and current_month < target_month):

                if current_month == 12:
                    current_month = 1
                    current_year += 1
                else:
                    current_month += 1

                next_sale = forecast_one_step(
                    category, product, current_year, current_month, sales_series
                )
                sales_series.append(next_sale)

            prediction = float(sales_series[-1])

            # Trend adjustment
            year_index = target_year - base_year
            trend_multiplier = 1 + (0.04 * year_index)
            detrended_prediction = prediction / trend_multiplier

            reference_value = get_reference_value(product_data)

            if reference_value == 0:
                demand_percentage = 0
            else:
                demand_percentage = (detrended_prediction / reference_value) * 100

            demand_percentage = round(min(demand_percentage, 100), 1)
            demand_level = classify_demand(demand_percentage)

        except Exception as e:
            error_message = str(e)

    context = {
        "prediction": prediction,
        "demand_level": demand_level,
        "demand_percentage": demand_percentage,
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


# =============================
# PEAK MONTH VIEW
# =============================

def peak_month(request):

    peak_month_name = None
    peak_month_num = None
    peak_demand_percentage = None
    peak_demand_level = None
    error_message = None

    selected_category = request.GET.get("category") or request.POST.get("category")
    selected_product = None
    selected_year = None

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

            selected_product = product
            selected_year = target_year

            product_data = df[df["Product Name"] == product].copy()
            product_data.sort_values(["Year", "Month"], inplace=True)

            sales_series = product_data["Sales"].tolist()

            # Advance series to end of (target_year - 1)
            temp_series = advance_series_to_base(
                category, product, target_year, product_data, sales_series
            )

            # Forecast all 12 months of target_year
            yearly_predictions = forecast_full_year(
                category, product, target_year, temp_series
            )

            # Trend adjustment
            year_index = target_year - base_year
            trend_multiplier = 1 + (0.04 * year_index)

            detrended_yearly = {
                m: (sales / trend_multiplier)
                for m, sales in yearly_predictions.items()
            }

            reference_value = get_reference_value(product_data)

            # Find peak month
            peak_month_num = max(detrended_yearly, key=detrended_yearly.get)
            peak_month_name = pd.Timestamp(
                target_year, peak_month_num, 1
            ).strftime('%B')

            peak_detrended = detrended_yearly[peak_month_num]

            if reference_value == 0:
                peak_demand_percentage = 0
            else:
                peak_demand_percentage = (peak_detrended / reference_value) * 100

            peak_demand_percentage = round(min(peak_demand_percentage, 100), 1)
            peak_demand_level = classify_demand(peak_demand_percentage)

        except Exception as e:
            error_message = str(e)

    context = {
        "categories": categories,
        "category_products": category_products,
        "selected_category": selected_category,
        "selected_product": selected_product,
        "selected_year": selected_year,
        "peak_month_name": peak_month_name,
        "peak_month_num": peak_month_num,
        "peak_demand_percentage": peak_demand_percentage,
        "peak_demand_level": peak_demand_level,
        "error_message": error_message,
    }

    return render(request, "peak_month.html", context)
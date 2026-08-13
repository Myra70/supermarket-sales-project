import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load CSV

df = pd.read_csv("/Users/apple/streamlit_project/SuperMarket Analysis - SuperMarket Analysis.csv")

print("Dataset loaded successfully!")
print(df.head())


# 2. Convert Date

df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


# 3. Convert Time

df["Hour"] = pd.to_datetime(
    df["Time"],
    format="%I:%M:%S %p"
).dt.hour

# 4. Select Features

features = [
    "Branch",
    "City",
    "Customer type",
    "Gender",
    "Product line",
    "Payment",
    "Quantity",
    "Month",
    "Day",
    "Hour"
]

X = df[features]


# Target = Sales
y = df["Sales"]


# 5. Categorical Features

categorical_features = [
    "Branch",
    "City",
    "Customer type",
    "Gender",
    "Product line",
    "Payment"
]


# 6. Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 7. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 8. Create Models

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


# Store results

results = []

best_model = None
best_r2 = float("-inf")
best_model_name = ""


# 9. Train Models

for name, algorithm in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                algorithm
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )


    # Evaluation

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


    print("\n--------------------------")
    print(name)
    print("--------------------------")

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


    # Find best model

    if r2 > best_r2:

        best_r2 = r2
        best_model = pipeline
        best_model_name = name


# 10. Save Best Model

joblib.dump(
    best_model,
    "model.pkl"
)


# 11. Save Results

results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_results.csv",
    index=False
)


print("\n==========================")
print("BEST MODEL")
print("==========================")

print(
    "Best Model:",
    best_model_name
)

print(
    "Best R2:",
    round(best_r2, 4)
)

print("\nmodel.pkl created successfully!")
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import polars as pl
import pandas as pd
import os


class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        """
        self.train_data = pl.read_csv(train_data_path)
        self.test_data = pl.read_csv(test_data_path)
        self.models = {}
        
        # Ensure outputs folder exists
        self.output_folder = "src/real_estate_toolkit/ml_models/outputs/"
        os.makedirs(self.output_folder, exist_ok=True)

    def clean_data(self):
        """
        Clean training and testing datasets.
        """
        # Drop columns with too many missing values or irrelevant ones
        self.train_data = self.train_data.drop_nulls(0.5)
        self.test_data = self.test_data.drop_nulls(0.5)

        # Fill missing values in numeric and categorical columns
        for col in self.train_data.columns:
            if self.train_data[col].dtype in [pl.Float64, pl.Int64]:
                self.train_data = self.train_data.with_column(
                    pl.col(col).fill_null(self.train_data[col].mean())
                )
                self.test_data = self.test_data.with_column(
                    pl.col(col).fill_null(self.train_data[col].mean())
                )
            elif self.train_data[col].dtype == pl.Categorical:
                self.train_data = self.train_data.with_column(
                    pl.col(col).fill_null("Missing")
                )
                self.test_data = self.test_data.with_column(
                    pl.col(col).fill_null("Missing")
                )

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare datasets for ML models.
        """
        # Convert to pandas for scikit-learn
        train_df = self.train_data.to_pandas()
        test_df = self.test_data.to_pandas()

        # Separate target and predictors
        y = train_df[target_column]
        if selected_predictors:
            X = train_df[selected_predictors]
        else:
            X = train_df.drop(columns=[target_column])

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Define pipelines for preprocessing
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, preprocessor

    def train_baseline_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate baseline models.
        """
        X_train, X_test, y_train, y_test, preprocessor = self.prepare_features()

        results = {}

        # Linear Regression
        lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
        lr_pipeline.fit(X_train, y_train)
        y_train_pred = lr_pipeline.predict(X_train)
        y_test_pred = lr_pipeline.predict(X_test)

        results["Linear Regression"] = {
            "metrics": {
                "MSE": mean_squared_error(y_test, y_test_pred),
                "MAE": mean_absolute_error(y_test, y_test_pred),
                "R2": r2_score(y_test, y_test_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_test_pred),
            },
            "model": lr_pipeline
        }
        self.models["LinearRegression"] = lr_pipeline

        # Random Forest Regressor
        rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))])
        rf_pipeline.fit(X_train, y_train)
        y_train_pred = rf_pipeline.predict(X_train)
        y_test_pred = rf_pipeline.predict(X_test)

        results["Random Forest"] = {
            "metrics": {
                "MSE": mean_squared_error(y_test, y_test_pred),
                "MAE": mean_absolute_error(y_test, y_test_pred),
                "R2": r2_score(y_test, y_test_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_test_pred),
            },
            "model": rf_pipeline
        }
        self.models["RandomForest"] = rf_pipeline

        return results

    def forecast_sales_price(self, model_type: str = 'LinearRegression'):
        """
        Forecast house prices on the test dataset using the trained model.
        """
        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} has not been trained.")

        model = self.models[model_type]

        # Prepare test data
        test_df = self.test_data.to_pandas()
        test_features = test_df.drop(columns=["SalePrice", "Id"], errors="ignore")

        # Predict using the model
        predictions = model.predict(test_features)

        # Create submission file
        submission = pd.DataFrame({
            "Id": test_df["Id"],
            "SalePrice": predictions
        })
        submission_file = os.path.join(self.output_folder, "submission.csv")
        submission.to_csv(submission_file, index=False)

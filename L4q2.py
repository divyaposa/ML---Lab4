#A2. Calculate MSE, RMSE, MAPE and R2 scores for the price prediction exercise done in Lab 02. 
# A2. Calculate MSE, RMSE, MAPE and R2 scores for the price prediction exercise.
# Analyse the results.

import pandas as pd  # For reading Excel files and working with DataFrames
import numpy as np  # For numerical computations
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score  # Regression metrics
import os  # To check if a file exists

# ---------------- Classes ----------------
class PurchaseDataLoader:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path  # Store the path to the Excel file

    def load_data(self):
        # Check if the Excel file exists at the given path
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found at: {self.excel_path}")  # Raise error if missing

        # Load the Excel file into a DataFrame
        df = pd.read_excel(self.excel_path)

        # Select input features for regression as a NumPy array
        X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values

        # Select target variable (Payment) as a NumPy array
        y = df['Payment (Rs)'].values

        return df, X, y  # Return the DataFrame, features, and target

class PricePredictor:
    def __init__(self):
        self.model = LinearRegression()  # Initialize the Linear Regression model

    def train_and_predict(self, X, y):
        self.model.fit(X, y)  # Train the model on the features and target
        return self.model.predict(X)  # Predict target values using the trained model

class RegressionEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        # Calculate Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Calculate Mean Absolute Percentage Error
        mape = mean_absolute_percentage_error(y_true, y_pred)

        # Calculate R-squared score
        r2 = r2_score(y_true, y_pred)

        # Print all metrics
        print("Regression Evaluation Metrics:")
        print(f"MSE  : {mse:.2f}")  # Print MSE rounded to 2 decimals
        print(f"RMSE : {rmse:.2f}")  # Print RMSE
        print(f"MAPE : {mape*100:.2f}%")  # Print MAPE as percentage
        print(f"R²   : {r2:.4f}")  # Print R² with 4 decimals

        return mse, rmse, mape, r2  # Return all metrics

# ---------------- Main ----------------
if __name__ == "__main__":
    EXCEL_PATH = r"C:/Users/Divya/Desktop/lab2dataset/Purchase data.xlsx"  # Path to the Excel file

    try:
        loader = PurchaseDataLoader(EXCEL_PATH)  # Create a data loader instance
        df, X, y = loader.load_data()  # Load the data

        predictor = PricePredictor()  # Create a predictor instance
        y_pred = predictor.train_and_predict(X, y)  # Train model and predict values

        evaluator = RegressionEvaluator()  # Create an evaluator instance
        evaluator.evaluate(y, y_pred)  # Calculate and print regression metrics

    except FileNotFoundError as e:
        print(f"Error: {e}")  # Print error message if file is not found

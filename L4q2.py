# Import required libraries
import pandas as pd  # For reading Excel files and handling tabular data with DataFrames
import numpy as np  # For numerical computations
from sklearn.linear_model import LinearRegression  # For building a linear regression model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score  # For evaluating regression performance
from typing import Tuple  # For defining function return types

# Class to load and preprocess purchase data from Excel
class PurchaseDataLoader:
    def __init__(self, excel_path: str, sheet_name: str):
        """
        Constructor that initializes the path to the Excel file and the sheet name.
        """
        self.excel_path = excel_path  # Store the path to the Excel file
        self.sheet_name = sheet_name  # Store the sheet name to load

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Loads the Excel file, extracts features and target variable.
        Returns:
            df: Full DataFrame with all data
            X: Numpy array of input features
            y: Numpy array of target (labels)
        """
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)  # Load data from Excel into a DataFrame
        X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values  # Extract input features as NumPy array
        y = df['Payment (Rs)'].values  # Extract the target variable (payment) as NumPy array
        return df, X, y  # Return the DataFrame and the features/labels

# Class to build and use a linear regression model
class PricePredictor:
    def __init__(self):
        """
        Constructor to initialize the Linear Regression model.
        """
        self.model = LinearRegression()  # Create an instance of scikit-learn's LinearRegression model

    def train_and_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Trains the model and makes predictions on the input features.
        Args:
            X: Input feature array
            y: Target output array
        Returns:
            predictions: Predicted output values
        """
        self.model.fit(X, y)  # Fit (train) the model using the input features and labels
        predictions = self.model.predict(X)  # Make predictions using the trained model
        return predictions  # Return the predicted values

# Class to evaluate how well the regression model performed
class RegressionEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates and prints regression metrics to evaluate model performance.
        Args:
            y_true: Actual values
            y_pred: Predicted values from the model
        Returns:
            mse, rmse, mape, r2: Evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)  # Calculate Mean Squared Error
        rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
        mape = mean_absolute_percentage_error(y_true, y_pred)  # Calculate Mean Absolute Percentage Error
        r2 = r2_score(y_true, y_pred)  # Calculate R-squared (coefficient of determination)

        # Print metrics in a readable format
        print("📊 Regression Evaluation Metrics:")
        print(f"MSE  : {mse:.2f}")  # Print MSE rounded to 2 decimals
        print(f"RMSE : {rmse:.2f}")  # Print RMSE
        print(f"MAPE : {mape*100:.2f}%")  # Print MAPE as percentage
        print(f"R²    : {r2:.4f}")  # Print R² with 4 decimals

        return mse, rmse, mape, r2  # Return all metrics

# ========== Main Execution Block ==========
if __name__ == "__main__":
    # Step 1: Specify path and sheet name of the Excel file to load
    EXCEL_PATH = r"C:/Users/Divya/Desktop/labdataset/Lab Session Data.xlsx"  # Path to the Excel file
    SHEET_NAME = "Purchase data"  # Name of the sheet containing the purchase data

    # Step 2: Load the dataset using the PurchaseDataLoader
    loader = PurchaseDataLoader(EXCEL_PATH, SHEET_NAME)  # Create an instance of the loader
    df, X, y = loader.load_data()  # Load the data (DataFrame, input features, and target)

    # Step 3: Train a model and generate predictions
    predictor = PricePredictor()  # Create the predictor object
    y_pred = predictor.train_and_predict(X, y)  # Train the model and get predictions

    # Step 4: Evaluate how well the model performed
    evaluator = RegressionEvaluator()  # Create the evaluator object
    evaluator.evaluate(y, y_pred)  # Print regression metrics for performance evaluation

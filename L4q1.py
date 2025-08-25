"""
A1. Please evaluate confusion matrix for your classification problem. From confusion matrix, the
other performance metrics such as precision, recall and F1-Score measures for both training and test
data. Based on your observations, infer the models learning outcome (underfit / regularfit / overfit). 
"""

# Import necessary libraries
import os  # To work with file paths
import numpy as np  # For arrays and math operations
import torch  # For handling image tensors
from torchvision import datasets, transforms  # To load and preprocess images
from sklearn.metrics import classification_report, confusion_matrix  # For metrics and confusion matrix
from sklearn.model_selection import train_test_split  # To split data into train and test
from sklearn.neighbors import KNeighborsClassifier  # KNN algorithm
import seaborn as sns  # For plotting heatmaps (visualize confusion matrix)
import matplotlib.pyplot as plt  # For general plotting
from tqdm import tqdm  # For showing progress bars

# Create a class to handle dataset, model, training, and evaluation
class MiniImageNetClassifier:
    def __init__(self, data_path, test_size=0.2, random_state=42, n_neighbors=3):
        """
        Initialize the classifier with dataset path and model parameters.
        """
        self.data_path = data_path  # Path where images are stored
        self.test_size = test_size  # Portion of data to use for testing
        self.random_state = random_state  # Seed to make results reproducible
        self.n_neighbors = n_neighbors  # Number of neighbors for KNN algorithm
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)  # Initialize KNN model

        # Placeholders for data and results
        self.X_train = None  # Training features
        self.X_test = None  # Test features
        self.y_train = None  # Training labels
        self.y_test = None  # Test labels
        self.y_train_pred = None  # Predictions on training set
        self.y_test_pred = None  # Predictions on test set

        self.class_names = []  # List to store class names

    def load_and_preprocess_data(self):
        """
        Loads images, flattens them, and splits them into training and test sets.
        """
        print("Loading and preprocessing data...")

        # Define transformations for images
        transform = transforms.Compose([
            transforms.Resize((84, 84)),  # Resize each image to 84x84 pixels
            transforms.ToTensor(),  # Convert image to a tensor
            transforms.Lambda(lambda x: x.view(-1))  # Flatten image into 1D vector
        ])

        # Load dataset from folder (expects folder per class)
        dataset = datasets.ImageFolder(self.data_path, transform=transform)
        self.class_names = dataset.classes  # Save class labels (folder names)

        X, y = [], []  # Lists to store images and labels

        # Loop through all images in dataset
        for img_tensor, label in tqdm(dataset, desc=" Processing images", total=len(dataset)):
            X.append(img_tensor)  # Add image tensor to features
            y.append(label)  # Add label

        # Convert lists to arrays for model
        X = torch.stack(X).numpy()  # Stack tensors and convert to NumPy array
        y = np.array(y)  # Convert labels to NumPy array

        # Split data into training and testing (stratified keeps class balance)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        print(f" Loaded {len(dataset)} samples. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def train_model(self):
        """
        Trains the KNN model on the training data.
        """
        print(" Training KNN model...")
        self.model.fit(self.X_train, self.y_train)  # Train model using training data
        print(" Training complete.")

    def evaluate_model(self):
        """
        Predicts and evaluates the model on both training and test data.
        """
        print("Evaluating model...")

        # Predict training and test labels
        self.y_train_pred = self._predict_with_progress(self.X_train, " Predicting train data")
        self.y_test_pred = self._predict_with_progress(self.X_test, " Predicting test data")

        # Show classification report (precision, recall, F1, accuracy) for training data
        print("\n==========  Train Classification Report ==========")
        print(classification_report(self.y_train, self.y_train_pred, target_names=self.class_names))

        # Show classification report for test data
        print("\n========== Test Classification Report ==========")
        print(classification_report(self.y_test, self.y_test_pred, target_names=self.class_names))

        # Compute confusion matrices
        cm_train = confusion_matrix(self.y_train, self.y_train_pred)
        cm_test = confusion_matrix(self.y_test, self.y_test_pred)

        # Plot confusion matrices as heatmaps
        self.plot_confusion_matrix(cm_train, "Train Confusion Matrix")
        self.plot_confusion_matrix(cm_test, "Test Confusion Matrix")

        # Analyze model fit (underfit, overfit, good fit)
        self.infer_model_fit()

    def _predict_with_progress(self, data, desc):
        """
        Predicts in batches and shows progress bar.
        """
        batch_size = 500  # Number of samples per batch
        preds = []  # List to store predictions

        # Loop through data in batches
        for i in tqdm(range(0, len(data), batch_size), desc=desc):
            batch = data[i:i + batch_size]  # Slice batch
            preds.extend(self.model.predict(batch))  # Predict and add to list

        return np.array(preds)  # Convert predictions to NumPy array

    def plot_confusion_matrix(self, cm, title):
        """
        Plot heatmap of confusion matrix.
        """
        plt.figure(figsize=(10, 8))  # Set figure size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # Plot values in each cell
                    xticklabels=self.class_names, yticklabels=self.class_names)  # Label axes
        plt.title(title)  # Set title
        plt.xlabel('Predicted')  # X-axis label
        plt.ylabel('True')  # Y-axis label
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.yticks(rotation=45)  # Rotate y-axis labels
        plt.tight_layout()  # Adjust layout
        plt.show()  # Display plot

    def infer_model_fit(self):
        """
        Check if model is underfit, overfit, or generalizes well.
        """
        # Calculate accuracy on train and test sets
        train_acc = np.mean(self.y_train == self.y_train_pred)
        test_acc = np.mean(self.y_test == self.y_test_pred)

        print(f"\n Train Accuracy: {train_acc:.2f}")  # Print train accuracy
        print(f" Test Accuracy:  {test_acc:.2f}")    # Print test accuracy

        # Simple rule to decide model fit
        if train_acc < 0.7 and test_acc < 0.7:
            print("UNDERFITTING: Model performs poorly on both training and test sets.")
        elif train_acc > 0.9 and test_acc < 0.7:
            print("OVERFITTING: Model performs well on training but poorly on test set.")
        else:
            print("GOOD FIT: Model generalizes well to unseen data.")

# ======== Run Classifier ========
if __name__ == "__main__":
    data_path = r"C:\Users\divya\Desktop\labdataset"  # Path to your dataset

    # Check if dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # Create classifier object
    classifier = MiniImageNetClassifier(data_path=data_path)

    # Load and preprocess data
    classifier.load_and_preprocess_data()

    # Train KNN model
    classifier.train_model()

    # Evaluate model (classification report + confusion matrix + fit check)
    classifier.evaluate_model()

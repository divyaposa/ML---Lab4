# Import necessary libraries
import os  # For file path handling
import numpy as np  # For numerical operations
import torch  # For tensor operations
from torchvision import datasets, transforms  # For image loading and preprocessing
from sklearn.metrics import classification_report, confusion_matrix  # For evaluation metrics
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
import seaborn as sns  # For heatmap visualization
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm  # For progress bars


class MiniImageNetClassifier:
    def __init__(self, data_path, test_size=0.2, random_state=42, n_neighbors=3):
        """
        Initialize the classifier with dataset path and model parameters.
        """
        self.data_path = data_path  # Dataset directory path
        self.test_size = test_size  # Proportion of test data
        self.random_state = random_state  # Seed for reproducibility
        self.n_neighbors = n_neighbors  # Number of neighbors for KNN
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)  # Create KNN model

        # Initialize placeholders for data and results
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_pred = None
        self.y_test_pred = None

        self.class_names = []  # Will store class labels

    def load_and_preprocess_data(self):
        """
        Loads images, flattens them, and splits them into training and test sets.
        """
        print("🔄 Loading and preprocessing data...")

        # Transformations: Resize image and flatten to 1D vector
        transform = transforms.Compose([
            transforms.Resize((84, 84)),  # Resize to 84x84 pixels
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Lambda(lambda x: x.view(-1))  # Flatten tensor to 1D
        ])

        # Load dataset using ImageFolder (requires folder/class structure)
        dataset = datasets.ImageFolder(self.data_path, transform=transform)
        self.class_names = dataset.classes  # Get list of class names

        X, y = [], []  # Initialize feature and label lists

        # Loop through dataset and collect image tensors and labels
        for img_tensor, label in tqdm(dataset, desc="📦 Processing images", total=len(dataset)):
            X.append(img_tensor)  # Add flattened image tensor
            y.append(label)  # Add label

        # Convert lists to NumPy arrays
        X = torch.stack(X).numpy()  # Stack tensors and convert to NumPy
        y = np.array(y)  # Convert labels to NumPy array

        # Split data into training and test sets with stratified sampling
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        # Print dataset summary
        print(f" Loaded {len(dataset)} samples. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def train_model(self):
        """
        Trains the KNN model on the training data.
        """
        print(" Training KNN model...")
        self.model.fit(self.X_train, self.y_train)  # Fit the model to training data
        print(" Training complete.")

    def evaluate_model(self):
        """
        Predicts and evaluates the model on both training and test data.
        """
        print("Evaluating model...")

        # Make predictions on train and test sets
        self.y_train_pred = self._predict_with_progress(self.X_train, "🔁 Predicting train data")
        self.y_test_pred = self._predict_with_progress(self.X_test, "🔁 Predicting test data")

        # Print classification report for training data
        print("\n========== 📊 Train Classification Report ==========")
        print(classification_report(self.y_train, self.y_train_pred, target_names=self.class_names))

        # Print classification report for testing data
        print("\n========== 📊 Test Classification Report ==========")
        print(classification_report(self.y_test, self.y_test_pred, target_names=self.class_names))

        # Compute confusion matrices
        cm_train = confusion_matrix(self.y_train, self.y_train_pred)
        cm_test = confusion_matrix(self.y_test, self.y_test_pred)

        # Plot confusion matrices
        self.plot_confusion_matrix(cm_train, "Train Confusion Matrix")
        self.plot_confusion_matrix(cm_test, "Test Confusion Matrix")

        # Analyze model performance
        self.infer_model_fit()

    def _predict_with_progress(self, data, desc):
        """
        Predicts in batches with a progress bar for large datasets.
        """
        batch_size = 500  # Number of samples per batch
        preds = []  # Store predictions

        # Loop through data in batches
        for i in tqdm(range(0, len(data), batch_size), desc=desc):
            batch = data[i:i + batch_size]  # Slice batch
            preds.extend(self.model.predict(batch))  # Predict and store results

        return np.array(preds)  # Return as NumPy array

    def plot_confusion_matrix(self, cm, title):
        """
        Plots a heatmap of the confusion matrix.
        """
        plt.figure(figsize=(10, 8))  # Set plot size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)  # Draw heatmap
        plt.title(title)  # Title of plot
        plt.xlabel('Predicted')  # X-axis label
        plt.ylabel('True')  # Y-axis label
        plt.xticks(rotation=45)  # Rotate x labels
        plt.yticks(rotation=45)  # Rotate y labels
        plt.tight_layout()  # Adjust layout
        plt.show()  # Display plot

    def infer_model_fit(self):
        """
        Checks for underfitting, overfitting, or good generalization.
        """
        # Calculate accuracy for train and test sets
        train_acc = np.mean(self.y_train == self.y_train_pred)
        test_acc = np.mean(self.y_test == self.y_test_pred)

        # Print accuracies
        print(f"\n Train Accuracy: {train_acc:.2f}")
        print(f" Test Accuracy:  {test_acc:.2f}")

        # Simple rule-based interpretation
        if train_acc < 0.7 and test_acc < 0.7:
            print("UNDERFITTING: Model performs poorly on both training and test sets.")
        elif train_acc > 0.9 and test_acc < 0.7:
            print("OVERFITTING: Model performs well on training but poorly on test set.")
        else:
            print("GOOD FIT: Model generalizes well to unseen data.")


# ======== Run Classifier ========
if __name__ == "__main__":
    # ✅ Provide the correct path to your MiniImageNet-style dataset (folder with subfolders as class names)
    data_path = r'C:/Users/Divya/Desktop/labdataset'

    # Check if dataset path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # Create classifier instance and run full pipeline
    classifier = MiniImageNetClassifier(data_path=data_path)
    classifier.load_and_preprocess_data()  # Load images and split data
    classifier.train_model()  # Train KNN
    classifier.evaluate_model()  # Evaluate and visualize

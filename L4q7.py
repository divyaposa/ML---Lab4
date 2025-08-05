import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # for progress bars

# ==========================================
# 💡 Class to extract image features using ResNet18
# ==========================================
class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # Load pre-trained ResNet18 model
        self.model = resnet18(pretrained=True)
        
        # Remove the final classification layer (keep only feature extractor)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval().to(self.device)

    def extract_features(self, dataloader):
        """
        Converts each image into a fixed-size feature vector using ResNet18
        """
        features = []
        labels = []

        with torch.no_grad():  # No need to calculate gradients
            for imgs, targets in tqdm(dataloader, desc="🔍 Extracting Features"):
                imgs = imgs.to(self.device)
                outputs = self.model(imgs).squeeze()  # Shape: [batch_size, 512]
                features.append(outputs.cpu().numpy())  # Move to CPU and NumPy
                labels.extend(targets.numpy())

        return np.vstack(features), np.array(labels)


# ==========================================
# 💡 Class to load miniImageNet data (only selected classes)
# ==========================================
class MiniImageNetDataLoader:
    def __init__(self, data_dir, selected_classes, samples_per_class=20):
        self.data_dir = data_dir
        self.selected_classes = selected_classes
        self.samples_per_class = samples_per_class

    def load_data(self):
        """
        Loads only selected classes from miniImageNet using ImageFolder.
        Limits number of samples per class.
        """
        transform = transforms.Compose([
            transforms.Resize((84, 84)),  # Resize all images to 84x84
            transforms.ToTensor()         # Convert image to tensor
        ])

        # Load all images from folder
        dataset = ImageFolder(self.data_dir, transform=transform)

        # Map class names to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

        # Select only samples from the two desired classes
        selected = {cls: [] for cls in self.selected_classes}
        for path, label in dataset.samples:
            cls = dataset.classes[label]
            if cls in self.selected_classes and len(selected[cls]) < self.samples_per_class:
                selected[cls].append((path, label))
        
        # Flatten selected data and update dataset
        final_samples = [item for sublist in selected.values() for item in sublist]
        dataset.samples = final_samples
        dataset.targets = [label for _, label in final_samples]

        # Return a PyTorch DataLoader for batch processing
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=16, shuffle=False)


# ==========================================
# 💡 Class to perform Hyperparameter Tuning for kNN
# ==========================================
class KNNHyperparameterTuner:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def tune_k(self, method='grid'):
        """
        Tries different values of 'k' (number of neighbors) and finds the best one using cross-validation.
        method: 'grid' or 'random'
        """
        # Try values of k from 1 to 20
        param_grid = {'n_neighbors': list(range(1, 21))}
        knn = KNeighborsClassifier()

        if method == 'grid':
            search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  # Tries every k
        elif method == 'random':
            search = RandomizedSearchCV(knn, param_grid, cv=5, n_iter=10, scoring='accuracy')  # Tries 10 random values
        else:
            raise ValueError("method must be 'grid' or 'random'")

        print(f"🔍 Running {method.capitalize()}SearchCV for kNN...")
        search.fit(self.X, self.y)  # Perform the search

        print(f"✅ Best k: {search.best_params_['n_neighbors']}")
        print(f"✅ Best Accuracy: {search.best_score_:.4f}")
        return search.best_params_['n_neighbors'], search.best_score_


# ==========================================
# 🚀 Main Execution
# ==========================================
if __name__ == "__main__":
    # 👉 Set your miniImageNet dataset path here
    DATA_PATH = r'C:/Users/Divya/Desktop/labdataset'
    
    # 👉 Choose any two classes from miniImageNet
    SELECTED_CLASSES = ['n01532829', 'n01749939']

    # Step 1: Load miniImageNet data with selected classes and limited samples
    loader = MiniImageNetDataLoader(DATA_PATH, SELECTED_CLASSES, samples_per_class=50)
    dataloader = loader.load_data()

    # Step 2: Extract deep features using pre-trained ResNet18
    extractor = FeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    X, y = extractor.extract_features(dataloader)

    # Step 3: Reduce features to 2D using PCA (Principal Component Analysis)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Step 4: Tune the 'k' hyperparameter for the kNN classifier
    tuner = KNNHyperparameterTuner(X_2d, y)
    best_k, best_score = tuner.tune_k(method='grid')  # Try method='random' too!
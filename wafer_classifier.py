"""
PyTorch Wafer Classification System
Builds on the WM-811K dataset analysis from the existing notebook
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from torchvision import transforms


class WaferDataset(Dataset):
    """Custom Dataset for wafer map data"""
    
    def __init__(self, wafer_maps: List[np.ndarray], labels: List[str], 
                 transform=None, target_size=(64, 64)):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.wafer_maps)
    
    def __getitem__(self, idx):
        wafer_map = self.wafer_maps[idx]
        label = self.encoded_labels[idx]
        
        # Resize wafer map to target size
        wafer_map = self._resize_wafer_map(wafer_map)
        
        # Convert to tensor and add channel dimension
        wafer_tensor = torch.FloatTensor(wafer_map).unsqueeze(0)
        
        if self.transform:
            wafer_tensor = self.transform(wafer_tensor)
            
        return wafer_tensor, torch.LongTensor([label]).squeeze()
    
    def _resize_wafer_map(self, wafer_map: np.ndarray) -> np.ndarray:
        """Resize wafer map to target size using interpolation"""
        from scipy.ndimage import zoom
        
        h, w = wafer_map.shape
        target_h, target_w = self.target_size
        
        zoom_h = target_h / h
        zoom_w = target_w / w
        
        return zoom(wafer_map, (zoom_h, zoom_w), order=1)


class WaferCNN(nn.Module):
    """Convolutional Neural Network for wafer classification"""
    
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (64, 64)):
        super(WaferCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class WaferClassifier:
    """Main classifier class that handles training and inference"""
    
    def __init__(self, model_params: Dict = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        self.model_params = model_params or {}
        
    def load_data(self, data_path: str = "local_data/LSWMD.pkl") -> Tuple[List, List]:
        """Load and preprocess wafer data from pickle file"""
        print("Loading wafer data...")
        
        try:
            df = pd.read_pickle(data_path)
            print(f"Loaded {len(df)} wafer samples")
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []
        
        # Extract labeled data (matching notebook logic)
        labeled_data = []
        labels = []
        
        for i, row in df.iterrows():
            try:
                # Check if both label and failure type exist and are not empty
                if row['trainTestLabel'] and row['failureType']:
                    train_label = row['trainTestLabel']
                    failure_type = row['failureType']
                    
                    if train_label and failure_type:  # Both must be non-empty
                        wafer_map = np.array(row['waferMap'])
                        labeled_data.append(wafer_map)
                        labels.append(failure_type)
            except (IndexError, TypeError):
                continue
        
        print(f"Extracted {len(labeled_data)} labeled samples")
        print(f"Failure types: {set(labels)}")
        
        return labeled_data, labels
    
    def prepare_datasets(self, wafer_maps: List, labels: List, 
                        test_size: float = 0.2, val_size: float = 0.1):
        """Split data into train/val/test sets"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            wafer_maps, labels, test_size=test_size, 
            stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            stratify=y_temp, random_state=42
        )
        
        # Create datasets
        train_dataset = WaferDataset(X_train, y_train)
        val_dataset = WaferDataset(X_val, y_val)
        test_dataset = WaferDataset(X_test, y_test)
        
        # Store label encoder
        self.label_encoder = train_dataset.label_encoder
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self, num_classes: int):
        """Create and initialize the model"""
        self.model = WaferCNN(num_classes).to(self.device)
        return self.model
    
    def train(self, train_dataset: WaferDataset, val_dataset: WaferDataset,
              batch_size: int = 32, epochs: int = 50, lr: float = 0.001):
        """Train the model"""
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=2)
        
        # Create model
        num_classes = len(self.label_encoder.classes_)
        self.create_model(num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        print(f"Training on {self.device}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Update learning rate
            scheduler.step()
            
            # Print progress for each epoch on one line
            print(f'Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def evaluate(self, test_dataset: WaferDataset, batch_size: int = 32):
        """Evaluate model on test set"""
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=2)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert back to original labels
        pred_labels = self.label_encoder.inverse_transform(all_predictions)
        true_labels = self.label_encoder.inverse_transform(all_targets)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels))
        
        return pred_labels, true_labels
    
    def predict(self, wafer_map: np.ndarray) -> str:
        """Predict failure type for a single wafer map"""
        self.model.eval()
        
        # Preprocess
        dataset = WaferDataset([wafer_map], ['dummy'])
        wafer_tensor, _ = dataset[0]
        wafer_tensor = wafer_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(wafer_tensor)
            _, predicted = torch.max(output, 1)
            
        predicted_label = self.label_encoder.inverse_transform([predicted.cpu().item()])[0]
        return predicted_label
    
    def save_model(self, path: str):
        """Save model and label encoder"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'model_params': self.model_params
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str, num_classes: int):
        """Load model and label encoder"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.create_model(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        self.model_params = checkpoint.get('model_params', {})
        
        print(f"Model loaded from {path}")


def main():
    """Main training pipeline"""
    # Initialize classifier
    classifier = WaferClassifier()
    
    # Load data
    wafer_maps, labels = classifier.load_data()
    
    if not wafer_maps:
        print("No data loaded. Please check the data path.")
        return
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(
        wafer_maps, labels
    )
    
    print(f"Dataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Train model
    history = classifier.train(
        train_dataset, val_dataset,
        batch_size=32,
        epochs=30,
        lr=0.001
    )
    
    # Evaluate on test set
    pred_labels, true_labels = classifier.evaluate(test_dataset)
    
    # Save model
    classifier.save_model('wafer_classifier_model.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracies'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == "__main__":
    main()

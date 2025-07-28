"""
Example usage of the Wafer Classification System
"""

from wafer_classifier import WaferClassifier
import numpy as np
import matplotlib.pyplot as plt

def quick_demo():
    """Quick demonstration of the wafer classifier"""
    
    print("=== Wafer Classification Demo ===\n")
    
    # Initialize classifier
    classifier = WaferClassifier()
    
    # Load data (you'll need the LSWMD.pkl file in local_data/)
    print("1. Loading wafer data...")
    wafer_maps, labels = classifier.load_data()
    
    if not wafer_maps:
        print("No data found. Please ensure LSWMD.pkl is in local_data/ directory")
        return
    
    print(f"Loaded {len(wafer_maps)} labeled samples")
    print(f"   Failure types found: {set(labels)}")
    
    # Show class distribution
    from collections import Counter
    class_counts = Counter(labels)
    print(f"\n   Class distribution:")
    for failure_type, count in class_counts.most_common():
        print(f"   - {failure_type}: {count} samples")
    
    # Prepare datasets
    print("\n2. Preparing train/validation/test splits...")
    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(
        wafer_maps, labels, test_size=0.2, val_size=0.1
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples") 
    print(f"   Test: {len(test_dataset)} samples")
    
    # Train model (reduced epochs for demo)
    print("\n3. Training model...")
    print("   (Using reduced epochs for demo - increase for better performance)")
    
    history = classifier.train(
        train_dataset, val_dataset,
        batch_size=16,  # Smaller batch for demo
        epochs=10,      # Reduced epochs for demo
        lr=0.001
    )
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    pred_labels, true_labels = classifier.evaluate(test_dataset)
    
    # Save model
    print("\n5. Saving model...")
    classifier.save_model('demo_wafer_model.pth')
    print("   Model saved as 'demo_wafer_model.pth'")
    
    # Demo prediction on a single sample
    print("\n6. Demo single prediction...")
    sample_wafer = wafer_maps[0]
    predicted_type = classifier.predict(sample_wafer)
    actual_type = labels[0]
    
    print(f"   Sample prediction: {predicted_type}")
    print(f"   Actual label: {actual_type}")
    print(f"   Match: {'Match' if predicted_type == actual_type else 'Does not match'}")
    
    print("\n=== Demo Complete ===")
    print("For full training, increase epochs to 50+ and batch_size to 32+")


def visualize_samples():
    """Visualize some wafer samples"""
    
    classifier = WaferClassifier()
    wafer_maps, labels = classifier.load_data()
    
    if not wafer_maps:
        print("No data found for visualization")
        return
    
    # Get one sample from each class
    unique_labels = list(set(labels))[:6]  # Show up to 6 classes
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, failure_type in enumerate(unique_labels):
        # Find first sample of this type
        idx = labels.index(failure_type)
        wafer_map = wafer_maps[idx]
        
        axes[i].imshow(wafer_map, cmap='viridis')
        axes[i].set_title(f'{failure_type}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(unique_labels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('wafer_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Wafer samples visualization saved as 'wafer_samples.png'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        visualize_samples()
    else:
        quick_demo()
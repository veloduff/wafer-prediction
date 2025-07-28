# Semiconductor Wafer Map Analysis & Classification System

In this repo:
- **Jupyter Notebook**: Analysis and visualization of wafer failure patterns
- **PyTorch System**: CNN-based classification pipeline for automated failure detection

## Prerequisite: WM-811K Wafer Map Dataset

The WM-811K (Wafer Map 811K) dataset, also known as the LSWMD (Large Semiconductor Wafer Map Dataset), 
is a comprehensive collection of semiconductor wafer testing data from real-world fabrication environments.

**Download the dataset**:
- Download the zip file (MIR-WM811K.zip) that has the `LSWMD.pkl` file from http://mirlab.org/dataSet/public/  
- Place pickle file in `local_data/` directory

### Dataset Overview

- **Source**: Real fabrication facilities of integrated circuit manufacturers
- **Publication**: Originally released by MIR Labs for academic research
- **Size**: Contains approximately 811,457 wafer maps
- **Labeled Data**: 172,950 wafers (21%) are labeled with failure patterns
- **Download**: Available at http://mirlab.org/dataSet/public/

### Failure Types Classification

1. **none** - No failure pattern (147,431 samples)
2. **Edge-Ring** - Ring pattern at wafer edge (9,680 samples)  
3. **Edge-Loc** - Localized edge failures (5,189 samples)
4. **Center** - Center-concentrated failures (4,294 samples)
5. **Loc** - Localized failure pattern (3,593 samples)
6. **Scratch** - Linear scratch patterns (1,193 samples)
7. **Random** - Randomly distributed failures (866 samples)
8. **Donut** - Ring-shaped failure pattern (555 samples)
9. **Near-full** - Almost complete wafer failure (149 samples)

Data analysis notebook: <a href="https://github.com/veloduff/wafer-prediction/blob/main/wafer_data_analysis.ipynb">wafer_data_analysis.ipynb</a>

The notebook provides:
- Dataset exploration and statistics
- Failure pattern visualization
- Data preprocessing techniques
- Statistical analysis of failure distributions


## Installation and setup for classification

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

1. **Train the model** with a smaller number epochs, batch size, and learning rate:
   ``` 
   batch_size=16
   epochs=10    
   lr=0.001
   ```

   Run `example_usage.py`:
   ```
   python example_usage.py
   ```

### Visualize Sample Wafer Maps

View sample wafer maps from each failure type:

```python
python example_usage.py viz
```

### Full Training Pipeline

```python
from wafer_classifier import WaferClassifier

# Initialize classifier
classifier = WaferClassifier()

# Load data
wafer_maps, labels = classifier.load_data()

# Prepare datasets
train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(
    wafer_maps, labels
)

# Train model
history = classifier.train(
    train_dataset, val_dataset,
    batch_size=32,
    epochs=50,
    lr=0.001
)

# Evaluate
pred_labels, true_labels = classifier.evaluate(test_dataset)

# Save model
classifier.save_model('wafer_model.pth')
```

### Single Wafer Prediction

```python
# Load trained model
classifier = WaferClassifier()
classifier.load_model('demo_wafer_model.pth', num_classes=9)

# Predict on new wafer map
predicted_type = classifier.predict(wafer_map_array)
print(f"Predicted failure type: {predicted_type}")
```



## Key Components

### WaferDataset
- Custom PyTorch Dataset for handling variable-sized wafer maps
- Automatic resizing to 64×64 pixels
- Label encoding for categorical failure types

### WaferCNN
- Convolutional neural network optimized for wafer pattern recognition
- Batch normalization for stable training
- Adaptive pooling for consistent feature map sizes

### WaferClassifier
- Main interface for training and inference
- Handles data loading, preprocessing, and model management
- Provides evaluation metrics and visualization

## File Structure

```
wafer-prediction/
├── README.md                    # This comprehensive documentation
├── wafer_data_analysis.ipynb    # Interactive analysis notebook
├── wafer_classifier.py          # Main PyTorch classification system
├── example_usage.py             # Demo and usage examples
├── demo_wafer_model.pth         # Pre-trained model file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── setup_data.py               # Data preparation utilities
└── local_data/
    └── LSWMD.pkl               # WM-811K dataset (download separately)
```

## Performance Considerations

- **Class Imbalance**: Dataset is heavily imbalanced (85% "none" class)
- **Hardware Requirements**: 
  - Minimum: CPU with 8GB RAM
  - Recommended: GPU with 4GB+ VRAM for faster training
  - Storage: ~2GB for dataset and models

## Training Tips

1. **Start Small**: Use the demo with reduced epochs first
2. **Monitor Overfitting**: Watch validation loss vs training loss
3. **Adjust Learning Rate**: Use learning rate scheduling
4. **Class Weights**: Consider weighted loss for imbalanced classes
5. **Data Augmentation**: Add rotations/flips for better generalization

## Research Applications

This system is valuable for:
- **Semiconductor Manufacturing**: Automated quality control and defect detection
- **Machine Learning Research**: Benchmark dataset for computer vision applications
- **Industrial AI**: Real-time wafer inspection systems
- **Academic Studies**: Semiconductor yield analysis and pattern recognition

## Extending the System

The modular design allows for easy extensions:
- **New Architectures**: Replace WaferCNN with ResNet, EfficientNet, etc.
- **Data Augmentation**: Add transforms to WaferDataset
- **Ensemble Methods**: Combine multiple models
- **Transfer Learning**: Use pre-trained vision models
- **Real-time Inference**: Deploy with TorchScript or ONNX

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch_size
2. **Poor Performance**: Increase epochs, adjust learning rate
3. **Data Loading Errors**: Check LSWMD.pkl file path and format
4. **Import Errors**: Ensure all requirements are installed
5. **Notebook Issues**: Ensure Jupyter is properly installed

## References

- WM-811K Dataset: [MIR Labs Dataset](http://mirlab.org/dataSet/public/)
- Original Paper: "A Benchmark Dataset for Semiconductor Wafer Map Analysis"
- PyTorch Documentation: [pytorch.org](https://pytorch.org/)

## License

This implementation is provided for educational and research purposes. Please cite the original WM-811K dataset authors when using this code.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the system.

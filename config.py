"""
Configuration file for Wafer Classification System
Adjust these parameters to tune model performance
"""

# Data Configuration
DATA_CONFIG = {
    'data_path': 'local_data/LSWMD.pkl',
    'target_size': (64, 64),  # Resize wafer maps to this size
    'test_size': 0.2,         # Fraction for test set
    'val_size': 0.1,          # Fraction for validation set
    'random_seed': 42,        # For reproducible splits
}

# Model Configuration
MODEL_CONFIG = {
    'input_channels': 1,      # Grayscale wafer maps
    'conv_channels': [32, 64, 128, 256],  # Channel progression
    'fc_layers': [512, 128],  # Fully connected layer sizes
    'dropout_rates': [0.5, 0.3],  # Dropout for FC layers
    'activation': 'relu',     # Activation function
    'use_batch_norm': True,   # Batch normalization
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,         # Batch size for training
    'epochs': 50,             # Number of training epochs
    'learning_rate': 0.001,   # Initial learning rate
    'weight_decay': 1e-4,     # L2 regularization
    'scheduler_step': 15,     # LR scheduler step size
    'scheduler_gamma': 0.5,   # LR decay factor
    'early_stopping_patience': 10,  # Early stopping patience
    'save_best_only': True,   # Save only best model
}

# Data Augmentation (optional)
AUGMENTATION_CONFIG = {
    'use_augmentation': False,  # Enable data augmentation
    'rotation_degrees': 90,     # Random rotation range
    'horizontal_flip': 0.5,     # Probability of horizontal flip
    'vertical_flip': 0.5,       # Probability of vertical flip
    'brightness': 0.1,          # Brightness adjustment range
    'contrast': 0.1,            # Contrast adjustment range
}

# Evaluation Configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'confusion_matrix': True,
    'classification_report': True,
    'per_class_metrics': True,
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_cuda': True,          # Use GPU if available
    'num_workers': 2,          # DataLoader workers
    'pin_memory': True,        # Pin memory for faster GPU transfer
    'mixed_precision': False,  # Use mixed precision training
}

# Paths Configuration
PATHS_CONFIG = {
    'model_save_dir': 'models/',
    'results_save_dir': 'results/',
    'plots_save_dir': 'plots/',
    'logs_save_dir': 'logs/',
}

# Class Information (from WM-811K dataset)
CLASS_INFO = {
    'failure_types': [
        'none',      # No failure pattern
        'Edge-Ring', # Ring pattern at wafer edge
        'Edge-Loc',  # Localized edge failures
        'Center',    # Center-concentrated failures
        'Loc',       # Localized failure pattern
        'Scratch',   # Linear scratch patterns
        'Random',    # Randomly distributed failures
        'Donut',     # Ring-shaped failure pattern
        'Near-full'  # Almost complete wafer failure
    ],
    'class_weights': None,  # Auto-calculate or specify weights for imbalanced classes
    'ignore_classes': [],   # Classes to ignore during training
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_to_file': True,
    'log_to_console': True,
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# Hyperparameter Search Configuration (for future use)
HYPERPARAM_CONFIG = {
    'search_method': 'grid',  # 'grid', 'random', 'bayesian'
    'param_ranges': {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64],
        'dropout_rates': [[0.3, 0.2], [0.5, 0.3], [0.7, 0.5]],
    },
    'n_trials': 20,  # For random/bayesian search
    'cv_folds': 3,   # Cross-validation folds
}

# Export all configurations
ALL_CONFIG = {
    'data': DATA_CONFIG,
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'augmentation': AUGMENTATION_CONFIG,
    'evaluation': EVAL_CONFIG,
    'hardware': HARDWARE_CONFIG,
    'paths': PATHS_CONFIG,
    'classes': CLASS_INFO,
    'logging': LOGGING_CONFIG,
    'hyperparams': HYPERPARAM_CONFIG,
}


def get_config(config_name=None):
    """
    Get configuration dictionary
    
    Args:
        config_name (str): Specific config to return ('data', 'model', etc.)
                          If None, returns all configurations
    
    Returns:
        dict: Configuration dictionary
    """
    if config_name is None:
        return ALL_CONFIG
    
    return ALL_CONFIG.get(config_name, {})


def update_config(config_name, updates):
    """
    Update specific configuration
    
    Args:
        config_name (str): Configuration section to update
        updates (dict): Dictionary of updates to apply
    """
    if config_name in ALL_CONFIG:
        ALL_CONFIG[config_name].update(updates)
    else:
        print(f"Warning: Configuration '{config_name}' not found")


# Quick access functions
def get_data_config():
    return DATA_CONFIG

def get_model_config():
    return MODEL_CONFIG

def get_training_config():
    return TRAINING_CONFIG


if __name__ == "__main__":
    # Print all configurations
    import json
    print("=== Wafer Classification Configuration ===")
    for section, config in ALL_CONFIG.items():
        print(f"\n{section.upper()}:")
        print(json.dumps(config, indent=2))
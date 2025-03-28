import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os
import logging
import platform
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----- M1 MAC GPU CONFIGURATION -----
def check_tensorflow_metal():
    """Check if tensorflow-metal is installed and properly configured"""
    # Check if we're on macOS
    if platform.system() != "Darwin":
        return False
    
    # Check if we're on Apple Silicon
    try:
        output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8')
        is_apple_silicon = "Apple" in output
    except:
        is_apple_silicon = platform.processor() == 'arm'
    
    if not is_apple_silicon:
        return False
    
    # Check TensorFlow version
    tf_version = tf.__version__
    logger.info(f"TensorFlow version: {tf_version}")
    
    # Print if Metal plugin is available
    try:
        import tensorflow_metal
        logger.info(f"tensorflow-metal plugin found: {tensorflow_metal.__version__}")
        return True
    except ImportError:
        logger.warning("tensorflow-metal package is not installed.")
        logger.warning("For M1/M2 Macs, install with:")
        logger.warning("pip install tensorflow==2.9.0")
        logger.warning("pip install tensorflow-metal==0.5.0")
        logger.warning("Note: Specific versions are required for compatibility.")
    
    # Set Metal environment variables
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_METAL_DEVICE_ENABLE'] = '1'
    
    # Try to detect Metal device
    devices = tf.config.list_physical_devices()
    logger.info(f"All detected devices: {devices}")
    
    return any('GPU' in str(device) or 'METAL' in str(device) for device in devices)

# Configure GPU for M1 Mac
def setup_gpu():
    # Configure Metal for M1 Mac
    is_metal_available = check_tensorflow_metal()

    if is_metal_available:
        logger.info("Metal acceleration configured and available.")
        # Try to limit memory growth for stability
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled.")
        except Exception as e:
            logger.warning(f"Could not configure GPU memory growth: {e}")
    else:
        logger.warning("Running on CPU. For faster training on M1/M2 Mac, install compatible versions:")
        logger.warning("pip install tensorflow==2.9.0")
        logger.warning("pip install tensorflow-metal==0.5.0")

    # Use mixed precision for faster training
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision policy set to: mixed_float16")
    except Exception as e:
        logger.warning(f"Could not set mixed precision policy: {e}")

# Custom callback for better logging during training
class LoggingCallback(keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        if epoch % 10 == 0:
            logger.info(f"{self.model_name} - Starting epoch {epoch+1}")
        
    def on_epoch_end(self, epoch, logs=None):
        if self.start_time is None:
            return
            
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        if epoch % 10 == 0 or epoch == self.params['epochs'] - 1:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            logger.info(f"{self.model_name} - Epoch {epoch+1}/{self.params['epochs']} - {metrics_str} - Time: {epoch_time:.2f}s")
    
    def on_train_end(self, logs=None):
        if self.epoch_times:
            avg_time = np.mean(self.epoch_times)
            logger.info(f"{self.model_name} - Training completed. Average time per epoch: {avg_time:.2f}s")

# Create model with different regularization techniques
def create_model(l1=0, l2=0, dropout_rate=0, nodes_per_layer=30, input_shape=None):
    logger.info(f"Creating model with L1={l1}, L2={l2}, Dropout={dropout_rate}, Nodes={nodes_per_layer}")
    
    model = keras.Sequential([
        layers.Dense(nodes_per_layer, activation="relu", 
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                    input_shape=input_shape),
        layers.Dropout(dropout_rate),
        layers.Dense(nodes_per_layer, activation="relu",
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    total_start_time = time.time()
    logger.info("Starting regularization techniques demonstration")
    
    # Set up GPU for M1 Mac
    setup_gpu()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. LOAD A REAL DATASET - CALIFORNIA HOUSING
    logger.info("Loading California Housing dataset...")
    load_start = time.time()
    
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split the data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Dataset loaded and processed in {time.time() - load_start:.2f}s")
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Features: {X_train.shape[1]}")

    # 3. TRAIN MODELS WITH DIFFERENT REGULARIZATION TECHNIQUES
    logger.info("Training models with different regularization techniques...")
    
    # Define the configurations to test
    models_config = [
        {"name": "Baseline (No Regularization)", "l1": 0, "l2": 0, "dropout": 0},
        {"name": "L1 Regularization", "l1": 0.01, "l2": 0, "dropout": 0},
        {"name": "L2 Regularization", "l2": 0.01, "l1": 0, "dropout": 0},
        {"name": "L1+L2 Regularization", "l1": 0.01, "l2": 0.01, "dropout": 0},
        {"name": "Dropout", "l1": 0, "l2": 0, "dropout": 0.3},
        {"name": "L2 + Dropout", "l1": 0, "l2": 0.01, "dropout": 0.3}
    ]

    # Train and evaluate each model
    histories = []
    val_scores = []
    epochs = 100

    for config in models_config:
        model_start = time.time()
        logger.info(f"\nTraining model: {config['name']}")
        
        model = create_model(
            l1=config['l1'], 
            l2=config['l2'], 
            dropout_rate=config['dropout'],
            input_shape=X_train.shape[1:]
        )
        
        # Add callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            LoggingCallback(config['name'])
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        histories.append((config['name'], history))
        
        # Evaluate on validation set
        val_score = model.evaluate(X_val_scaled, y_val, verbose=0)
        val_scores.append((config['name'], val_score))
        
        logger.info(f"  Model: {config['name']}")
        logger.info(f"  Training time: {time.time() - model_start:.2f}s")
        logger.info(f"  Validation MAE: {val_score[1]:.4f}")

    # 4. VISUALIZE THE RESULTS
    logger.info("Plotting learning curves for each model...")
    
    # Plot training and validation learning curves
    plt.figure(figsize=(15, 10))

    for i, (name, history) in enumerate(histories):
        plt.subplot(2, 3, i+1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.ion()
    plt.draw()
    plt.pause(0.001)

    # Plot validation MAE comparison
    logger.info("Plotting validation MAE comparison...")
    plt.figure(figsize=(12, 6))
    names = [config['name'] for config in models_config]
    maes = [score[1] for _, score in val_scores]

    plt.bar(names, maes)
    plt.title('Validation MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.ion()
    plt.draw()
    plt.pause(0.001)

    # 5. EXPLORE DIFFERENT MODEL CAPACITIES
    logger.info("Exploring effect of model capacity...")
    capacity_start = time.time()
    
    # Create models with different capacities with and without regularization
    capacities = [5, 30, 100, 300]
    capacity_results = []

    for nodes in capacities:
        nodes_start = time.time()
        logger.info(f"Testing capacity with {nodes} nodes per layer")
        
        # Without regularization
        model_no_reg = create_model(
            nodes_per_layer=nodes,
            input_shape=X_train.shape[1:]
        )
        history_no_reg = model_no_reg.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                LoggingCallback(f"No Reg - {nodes} nodes")
            ],
            verbose=0
        )
        
        # With regularization (L2 + dropout)
        model_reg = create_model(
            l2=0.01, 
            dropout_rate=0.3, 
            nodes_per_layer=nodes,
            input_shape=X_train.shape[1:]
        )
        history_reg = model_reg.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                LoggingCallback(f"With Reg - {nodes} nodes")
            ],
            verbose=0
        )
        
        # Evaluate
        score_no_reg = model_no_reg.evaluate(X_val_scaled, y_val, verbose=0)
        score_reg = model_reg.evaluate(X_val_scaled, y_val, verbose=0)
        
        capacity_results.append({
            "nodes": nodes,
            "no_reg_mae": score_no_reg[1],
            "reg_mae": score_reg[1],
            "no_reg_history": history_no_reg,
            "reg_history": history_reg
        })
        
        logger.info(f"  Nodes: {nodes} - Time: {time.time() - nodes_start:.2f}s")
        logger.info(f"    No Regularization MAE: {score_no_reg[1]:.4f}")
        logger.info(f"    Regularization MAE: {score_reg[1]:.4f}")
    
    logger.info(f"Capacity exploration completed in {time.time() - capacity_start:.2f}s")

    # Visualize the impact of model capacity
    logger.info("Plotting capacity impact analysis...")
    plt.figure(figsize=(12, 5))

    # Plot MAE vs. Capacity
    plt.subplot(1, 2, 1)
    plt.plot(
        [r["nodes"] for r in capacity_results],
        [r["no_reg_mae"] for r in capacity_results],
        'o-', label='No Regularization'
    )
    plt.plot(
        [r["nodes"] for r in capacity_results],
        [r["reg_mae"] for r in capacity_results],
        'o-', label='With Regularization'
    )
    plt.title('Effect of Model Capacity on Validation MAE')
    plt.xlabel('Neurons per Layer')
    plt.ylabel('Validation MAE')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot the benefit of regularization at different capacities
    plt.subplot(1, 2, 2)
    improvement = [
        (r["no_reg_mae"] - r["reg_mae"]) / r["no_reg_mae"] * 100
        for r in capacity_results
    ]
    plt.bar([str(r["nodes"]) for r in capacity_results], improvement)
    plt.title('Regularization Benefit by Model Capacity')
    plt.xlabel('Neurons per Layer')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.ion()
    plt.draw()
    plt.pause(0.001)

    # 6. EARLY STOPPING DEMONSTRATION
    logger.info("Running early stopping demonstration...")
    early_stopping_start = time.time()
    
    # Train a model with many epochs and plot the curves
    model = create_model(
        nodes_per_layer=100,
        input_shape=X_train.shape[1:]
    )
    full_history = model.fit(
        X_train_scaled, y_train,
        epochs=300,  # Train for many epochs
        validation_data=(X_val_scaled, y_val),
        callbacks=[LoggingCallback("Early Stopping Demo")],
        verbose=0
    )

    # Find the best epoch
    best_epoch = np.argmin(full_history.history['val_loss']) + 1
    logger.info(f"Best epoch found at: {best_epoch}")
    logger.info(f"Early stopping demonstration completed in {time.time() - early_stopping_start:.2f}s")

    # Plot learning curves with early stopping point
    logger.info("Plotting early stopping visualization...")
    plt.figure(figsize=(10, 6))
    plt.plot(full_history.history['loss'], label='Training Loss')
    plt.plot(full_history.history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', 
              label=f'Best Epoch ({best_epoch})')

    # Mark the three zones
    underfitting_end = best_epoch // 2
    overfitting_start = best_epoch

    plt.axvspan(0, underfitting_end, alpha=0.2, color='blue', label='Underfitting Zone')
    plt.axvspan(underfitting_end, overfitting_start, alpha=0.2, color='green', label='Optimal Fitting Zone')
    plt.axvspan(overfitting_start, 300, alpha=0.2, color='red', label='Overfitting Zone')

    plt.title('Early Stopping Demonstration')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    
    total_time = time.time() - total_start_time
    logger.info(f"Regularization techniques demo completed in {total_time:.2f} seconds")
    
    # Keep plots open until user closes them
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
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
        # Continue anyway as Metal might still work without explicit package in newer TF versions
    
    # Set Metal environment variables
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_METAL_DEVICE_ENABLE'] = '1'
    
    # Try to detect Metal device
    devices = tf.config.list_physical_devices()
    logger.info(f"All detected devices: {devices}")
    
    return any('GPU' in str(device) or 'METAL' in str(device) for device in devices)

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

# ----- PERFORMANCE CONFIGURATION -----
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
        if epoch % 5 == 0:
            logger.info(f"{self.model_name} - Starting epoch {epoch+1}")
        
    def on_epoch_end(self, epoch, logs=None):
        if self.start_time is None:
            return
            
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        if epoch % 5 == 0 or epoch == self.params['epochs'] - 1:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            logger.info(f"{self.model_name} - Epoch {epoch+1}/{self.params['epochs']} - {metrics_str} - Time: {epoch_time:.2f}s")
    
    def on_train_end(self, logs=None):
        if self.epoch_times:
            avg_time = np.mean(self.epoch_times)
            logger.info(f"{self.model_name} - Training completed. Average time per epoch: {avg_time:.2f}s")

# Function to train a model with given number of layers and neurons
def create_and_train_model(num_layers, neurons, dropout_rate=0, l2_reg=0, epochs=100, model_name="Model"):
    logger.info(f"Creating {model_name} - Layers: {num_layers}, Neurons: {neurons}, Dropout: {dropout_rate}, L2: {l2_reg}")
    
    start_time = time.time()
    
    # Create model with explicit input shape
    model = keras.Sequential()
    
    # First layer needs input_shape
    model.add(layers.Dense(neurons, activation='relu', 
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
                         input_shape=(1,)))  # Explicit input shape for first layer
    
    # Add remaining layers
    for i in range(1, num_layers):
        model.add(layers.Dense(neurons, activation='relu',
                             kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))
        
        # Add dropout if specified
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Build the model explicitly
    model.build((None, 1))
    
    # Compile model with best device strategy
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Now it's safe to count parameters
    logger.info(f"{model_name} - Model compiled with {model.count_params()} parameters")
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        LoggingCallback(model_name)
    ]
    
    logger.info(f"{model_name} - Starting training for up to {epochs} epochs (batch size: 16)")
    # Train model with early stopping
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=0,  # Disable default TF verbosity since we're using custom callback
        batch_size=16,
        callbacks=callbacks
    )
    
    # Calculate final metrics
    train_metrics = model.evaluate(x_train, y_train, verbose=0)
    val_metrics = model.evaluate(x_val, y_val, verbose=0)
    
    logger.info(f"{model_name} - Training completed in {time.time() - start_time:.2f}s")
    logger.info(f"{model_name} - Final training loss: {train_metrics[0]:.4f}, MAE: {train_metrics[1]:.4f}")
    logger.info(f"{model_name} - Final validation loss: {val_metrics[0]:.4f}, MAE: {val_metrics[1]:.4f}")
    
    return model, history

# Visualize the learning curves
def plot_history(histories, labels):
    logger.info("Plotting learning curves...")
    start_time = time.time()
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for i, (history, label) in enumerate(zip(histories, labels)):
        plt.plot(history.history['loss'], label=f'{label} (Training)')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    for i, (history, label) in enumerate(zip(histories, labels)):
        plt.plot(history.history['val_loss'], label=f'{label} (Validation)')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Use non-blocking mode for plotting to avoid hang
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    
    logger.info(f"Learning curves plotted in {time.time() - start_time:.2f}s")

# Make predictions and visualize results
def plot_predictions(models, labels):
    logger.info("Generating predictions and plotting results...")
    start_time = time.time()
    
    # Generate points for smooth curve
    x_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the original data
    plt.scatter(x_train, y_train, alpha=0.4, label='Training data')
    plt.scatter(x_val, y_val, alpha=0.4, label='Validation data')
    
    # Plot predictions for each model
    for model, label in zip(models, labels):
        pred_start = time.time()
        logger.info(f"Generating predictions for {label} model...")
        y_pred = model.predict(x_test, verbose=0)
        logger.info(f"Predictions generated in {time.time() - pred_start:.2f}s")
        plt.plot(x_test, y_pred, linewidth=2, label=f'{label} prediction')
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use non-blocking mode for plotting to avoid hang
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    
    logger.info(f"Predictions plot completed in {time.time() - start_time:.2f}s")

def main():
    global x_train, y_train, x_val, y_val
    
    logger.info("Starting machine learning fundamentals demo")
    total_start_time = time.time()
    
    # 1. GENERATING SYNTHETIC DATA
    logger.info("Generating synthetic dataset...")
    # Create a non-linear function with some noise
    x = np.linspace(-3, 3, 100)
    y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=len(x))
    logger.info(f"Generated {len(x)} data points with quadratic function")

    # Split into training and validation sets
    indices = np.random.permutation(len(x))
    train_idx, val_idx = indices[:70], indices[70:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # Reshape for Keras
    x_train = x_train.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)
    logger.info(f"Data split complete - Training: {len(x_train)}, Validation: {len(x_val)}")

    # 2. VISUALIZE OUR DATA
    logger.info("Visualizing dataset...")
    viz_start = time.time()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.7, label='Training data')
    plt.scatter(x_val, y_val, alpha=0.7, label='Validation data')
    plt.title('Synthetic Regression Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use non-blocking mode for plotting
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    
    logger.info(f"Dataset visualization completed in {time.time() - viz_start:.2f}s")
    logger.info(f"Training set size: {len(x_train)} samples")
    logger.info(f"Validation set size: {len(x_val)} samples")

    # 3. EXPERIMENT WITH MODEL CAPACITY: UNDERFITTING VS OVERFITTING
    logger.info("Starting model experiments...")

    # 4. EXPERIMENT 1: UNDERFITTING (VERY SIMPLE MODEL)
    underfit_model, underfit_history = create_and_train_model(
        num_layers=1, 
        neurons=2, 
        model_name="Underfit Model",
        epochs=50  # Reduced epochs for faster execution
    )

    # 5. EXPERIMENT 2: GOOD FIT
    good_model, good_history = create_and_train_model(
        num_layers=2, 
        neurons=16, 
        model_name="Good Fit Model",
        epochs=50  # Reduced epochs for faster execution
    )

    # 6. EXPERIMENT 3: OVERFITTING (VERY COMPLEX MODEL)
    overfit_model, overfit_history = create_and_train_model(
        num_layers=4, 
        neurons=64, 
        model_name="Overfit Model",
        epochs=50  # Reduced epochs for faster execution
    )

    # 7. EXPERIMENT 4: REGULARIZED MODEL (L2)
    l2_model, l2_history = create_and_train_model(
        num_layers=4, 
        neurons=64, 
        l2_reg=0.01, 
        model_name="L2 Regularized Model",
        epochs=50  # Reduced epochs for faster execution
    )

    # 8. EXPERIMENT 5: REGULARIZED MODEL (DROPOUT)
    dropout_model, dropout_history = create_and_train_model(
        num_layers=4, 
        neurons=64, 
        dropout_rate=0.3, 
        model_name="Dropout Regularized Model",
        epochs=50  # Reduced epochs for faster execution
    )

    # 9. VISUALIZE THE LEARNING CURVES
    plot_history(
        [underfit_history, good_history, overfit_history, l2_history, dropout_history],
        ['Underfit', 'Good Fit', 'Overfit', 'L2 Regularized', 'Dropout']
    )

    # 10. MAKE PREDICTIONS AND VISUALIZE RESULTS
    plot_predictions(
        [underfit_model, good_model, overfit_model, l2_model, dropout_model],
        ['Underfit', 'Good Fit', 'Overfit', 'L2 Regularized', 'Dropout']
    )
    
    total_time = time.time() - total_start_time
    logger.info(f"Demo completed in {total_time:.2f} seconds")
    
    # Keep plots open until user closes them
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
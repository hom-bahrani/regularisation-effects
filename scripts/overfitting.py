import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU/Metal availability on M1 Mac
logger.info("Checking for GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPU is available: {gpus}")
    # Enable memory growth to prevent TF from allocating all GPU memory at once
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled.")
    except RuntimeError as e:
        logger.warning(f"Error setting memory growth: {e}")
else:
    logger.info("No GPU found. Running on CPU.")

# For M1 Mac, Apple's Metal plugin for TensorFlow might be available
try:
    # Check if we're on macOS and if Metal is available
    if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'get_visible_devices'):
        physical_devices = tf.config.experimental.get_visible_devices('GPU')
        if len(physical_devices) > 0:
            logger.info(f"Metal GPU acceleration is available: {physical_devices}")
            # Configure TensorFlow to use the Metal plugin
            os.environ['TF_METAL_DEVICE_ENABLE'] = '1'
            logger.info("Metal acceleration enabled.")
except Exception as e:
    logger.warning(f"Error checking Metal availability: {e}")

# Custom callback for better logging during training
class LoggingCallback(keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        if epoch % 10 == 0:
            logger.info(f"{self.model_name} - Starting epoch {epoch+1}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        if epoch % 10 == 0 or epoch == self.params['epochs'] - 1:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            logger.info(f"{self.model_name} - Epoch {epoch+1}/{self.params['epochs']} - {metrics_str} - Time: {epoch_time:.2f}s")
    
    def on_train_end(self, logs=None):
        avg_time = np.mean(self.epoch_times)
        logger.info(f"{self.model_name} - Training completed. Average time per epoch: {avg_time:.2f}s")

# Function to train a model with given number of layers and neurons
def create_and_train_model(num_layers, neurons, dropout_rate=0, l2_reg=0, epochs=100, model_name="Model"):
    logger.info(f"Creating {model_name} - Layers: {num_layers}, Neurons: {neurons}, Dropout: {dropout_rate}, L2: {l2_reg}")
    
    start_time = time.time()
    layers_list = []
    
    # Input layer
    for i in range(num_layers):
        if i == 0:
            # First layer
            layers_list.append(layers.Dense(neurons, activation='relu', 
                                           kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))
            logger.debug(f"Added input layer with {neurons} neurons" + 
                      (f" and L2 regularization {l2_reg}" if l2_reg > 0 else ""))
        else:
            # Hidden layers
            layers_list.append(layers.Dense(neurons, activation='relu',
                                           kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))
            logger.debug(f"Added hidden layer {i} with {neurons} neurons" + 
                      (f" and L2 regularization {l2_reg}" if l2_reg > 0 else ""))
        
        # Add dropout if specified
        if dropout_rate > 0:
            layers_list.append(layers.Dropout(dropout_rate))
            logger.debug(f"Added dropout layer with rate {dropout_rate}")
    
    # Output layer
    layers_list.append(layers.Dense(1))
    logger.debug("Added output layer with 1 neuron")
    
    # Create model
    model = keras.Sequential(layers_list)
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logger.info(f"{model_name} - Model compiled with {model.count_params()} parameters")
    
    # Log model summary to debug level
    model.summary(print_fn=logger.debug)
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        LoggingCallback(model_name)
    ]
    
    logger.info(f"{model_name} - Starting training for up to {epochs} epochs")
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
    plt.show()
    logger.info("Learning curves plotted")

# Make predictions and visualize results
def plot_predictions(models, labels):
    logger.info("Generating predictions and plotting results...")
    
    # Generate points for smooth curve
    x_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the original data
    plt.scatter(x_train, y_train, alpha=0.4, label='Training data')
    plt.scatter(x_val, y_val, alpha=0.4, label='Validation data')
    
    # Plot predictions for each model
    for model, label in zip(models, labels):
        logger.info(f"Generating predictions for {label} model...")
        y_pred = model.predict(x_test, verbose=0)
        plt.plot(x_test, y_pred, linewidth=2, label=f'{label} prediction')
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    logger.info("Prediction comparison plotted")

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
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.7, label='Training data')
    plt.scatter(x_val, y_val, alpha=0.7, label='Validation data')
    plt.title('Synthetic Regression Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    logger.info(f"Training set size: {len(x_train)} samples")
    logger.info(f"Validation set size: {len(x_val)} samples")

    # 3. EXPERIMENT WITH MODEL CAPACITY: UNDERFITTING VS OVERFITTING
    logger.info("Starting model experiments...")

    # 4. EXPERIMENT 1: UNDERFITTING (VERY SIMPLE MODEL)
    underfit_model, underfit_history = create_and_train_model(
        num_layers=1, 
        neurons=2, 
        model_name="Underfit Model"
    )

    # 5. EXPERIMENT 2: GOOD FIT
    good_model, good_history = create_and_train_model(
        num_layers=2, 
        neurons=16, 
        model_name="Good Fit Model"
    )

    # 6. EXPERIMENT 3: OVERFITTING (VERY COMPLEX MODEL)
    overfit_model, overfit_history = create_and_train_model(
        num_layers=4, 
        neurons=64, 
        model_name="Overfit Model"
    )

    # 7. EXPERIMENT 4: REGULARIZED MODEL (L2)
    l2_model, l2_history = create_and_train_model(
        num_layers=4, 
        neurons=64, 
        l2_reg=0.01, 
        model_name="L2 Regularized Model"
    )

    # 8. EXPERIMENT 5: REGULARIZED MODEL (DROPOUT)
    dropout_model, dropout_history = create_and_train_model(
        num_layers=4, 
        neurons=64, 
        dropout_rate=0.3, 
        model_name="Dropout Regularized Model"
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

if __name__ == "__main__":
    main()
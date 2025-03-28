import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
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

# 3. MODEL CREATION FUNCTION
def create_model(neurons=16):
    model = keras.Sequential([
        layers.Dense(neurons, activation='relu', input_shape=(1,)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def build_keras_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Iterated K-fold with shuffling
def iterated_kfold(X, y, n_iterations=3, n_splits=5):
    logger.info(f"Running iterated K-fold validation with {n_iterations} iterations and {n_splits} splits")
    start_time = time.time()
    all_scores = []
    
    for iteration in range(n_iterations):
        logger.info(f"Iteration {iteration+1}/{n_iterations}")
        
        # Create a new KFold with different shuffling
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42+iteration)
        
        iteration_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_start = time.time()
            # Split data
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Scale the data
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            # Train the model
            model = build_keras_model()
            model.fit(X_fold_train_scaled, y_fold_train, epochs=100, verbose=0)
            
            # Evaluate the model
            y_pred = model.predict(X_fold_val_scaled, verbose=0)
            mse = mean_squared_error(y_fold_val, y_pred)
            iteration_scores.append(mse)
            
            logger.info(f"  Fold {fold+1}/{n_splits} - MSE: {mse:.4f} - Time: {time.time() - fold_start:.2f}s")
        
        all_scores.append(iteration_scores)
        logger.info(f"Iteration {iteration+1} completed - Avg MSE: {np.mean(iteration_scores):.4f}")
    
    logger.info(f"Iterated K-fold completed in {time.time() - start_time:.2f}s")
    return np.array(all_scores)

def main():
    total_start_time = time.time()
    logger.info("Starting cross-validation demonstration")
    
    # Set up GPU for M1 Mac
    setup_gpu()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. GENERATE MORE SYNTHETIC DATA
    logger.info("Generating synthetic dataset...")
    n_samples = 200
    x = np.linspace(-5, 5, n_samples)
    y = 0.5 * x**2 + 0.5 * x + np.sin(x) + np.random.normal(0, 1, size=len(x))
    logger.info(f"Generated {n_samples} data points with quadratic function")

    # Let's visualize the data
    logger.info("Visualizing dataset...")
    viz_start = time.time()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title('Synthetic Regression Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    
    # Use non-blocking mode for plotting
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    
    logger.info(f"Dataset visualization completed in {time.time() - viz_start:.2f}s")

    # 2. SIMPLE HOLDOUT VALIDATION
    logger.info("Performing simple holdout validation...")
    X_train, X_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y, test_size=0.2, random_state=42
    )

    logger.info(f"Training set size: {len(X_train)} samples")
    logger.info(f"Test set size: {len(X_test)} samples")

    # 4. SIMPLE TRAINING WITH HOLDOUT VALIDATION
    logger.info("Training model with holdout validation...")
    holdout_start = time.time()
    
    X_train_holdout, X_val_holdout, y_train_holdout, y_val_holdout = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model_holdout = create_model()
    model_holdout.summary(print_fn=logger.debug)
    
    callbacks = [LoggingCallback("Holdout Model")]
    
    history = model_holdout.fit(
        X_train_holdout, y_train_holdout,
        epochs=100,
        validation_data=(X_val_holdout, y_val_holdout),
        verbose=0,
        callbacks=callbacks
    )
    
    # Evaluate
    holdout_score = model_holdout.evaluate(X_val_holdout, y_val_holdout, verbose=0)
    logger.info(f"Holdout validation - Loss: {holdout_score:.4f}")
    logger.info(f"Holdout validation completed in {time.time() - holdout_start:.2f}s")

    # Plot the learning curves
    logger.info("Plotting holdout learning curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves with Simple Holdout Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ion()
    plt.draw()
    plt.pause(0.001)

    # 5. K-FOLD CROSS VALIDATION
    logger.info("Setting up K-fold cross validation...")
    # Create a Keras Regressor
    keras_regressor = KerasRegressor(
        build_fn=build_keras_model,
        epochs=100,
        batch_size=16,
        verbose=0
    )

    # Create a pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', keras_regressor)
    ])

    # Define K-fold cross-validation
    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Perform K-fold cross-validation
    logger.info(f"Performing {k}-fold cross-validation...")
    kfold_start = time.time()
    
    scores = []
    fold_predictions = []
    fold_actual = []

    # Manual K-fold implementation for visualization
    for i, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        fold_start = time.time()
        logger.info(f"Training fold {i+1}/{k}")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Scale the data
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)
        
        # Train the model
        model = build_keras_model()
        model.fit(X_fold_train_scaled, y_fold_train, epochs=100, verbose=0)
        
        # Evaluate the model
        y_pred = model.predict(X_fold_val_scaled, verbose=0)
        mse = mean_squared_error(y_fold_val, y_pred)
        scores.append(mse)
        
        # Save predictions for visualization
        fold_predictions.append((X_fold_val, y_pred))
        fold_actual.append((X_fold_val, y_fold_val))
        
        logger.info(f"  Fold {i+1} - MSE: {mse:.4f} - Time: {time.time() - fold_start:.2f}s")

    logger.info(f"K-fold cross-validation completed in {time.time() - kfold_start:.2f}s")
    logger.info(f"K-fold cross-validation scores (MSE): {[f'{s:.4f}' for s in scores]}")
    logger.info(f"Average MSE: {np.mean(scores):.4f}")
    logger.info(f"Standard deviation: {np.std(scores):.4f}")

    # 6. VISUALIZE THE K-FOLD RESULTS
    logger.info("Visualizing K-fold results...")
    plt.figure(figsize=(15, 10))

    for i in range(k):
        plt.subplot(2, 3, i+1)
        
        # Get the data for this fold
        X_val, y_val_true = fold_actual[i]
        _, y_val_pred = fold_predictions[i]
        
        # Plot the data
        plt.scatter(X_val, y_val_true, alpha=0.7, label='Actual')
        plt.scatter(X_val, y_val_pred, alpha=0.7, label='Predicted')
        
        plt.title(f'Fold {i+1} (MSE: {scores[i]:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    plt.bar(range(1, k+1), scores)
    plt.axhline(np.mean(scores), color='r', linestyle='--', 
                label=f'Mean MSE: {np.mean(scores):.4f}')
    plt.title('MSE Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.ion()
    plt.draw()
    plt.pause(0.001)

    # 7. ITERATED K-FOLD WITH SHUFFLING
    logger.info("Running iterated K-fold with shuffling...")
    iterations = 2  # Reduced for demonstration purposes
    iterated_scores = iterated_kfold(X_train, y_train, n_iterations=iterations)
    
    plt.figure(figsize=(12, 6))
    for i, scores in enumerate(iterated_scores):
        plt.plot(range(1, len(scores)+1), scores, 'o-', label=f'Iteration {i+1}')

    plt.axhline(np.mean(iterated_scores), color='r', linestyle='--', 
                label=f'Overall Mean: {np.mean(iterated_scores):.4f}')
    plt.title('Iterated K-Fold Cross-Validation Results')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    
    total_time = time.time() - total_start_time
    logger.info(f"Cross-validation demo completed in {total_time:.2f} seconds")
    
    # Keep plots open until user closes them
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
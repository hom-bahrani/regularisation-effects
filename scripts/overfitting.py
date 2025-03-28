import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to train a model with given number of layers and neurons
def create_and_train_model(num_layers, neurons, dropout_rate=0, l2_reg=0, epochs=100):
    layers_list = []
    
    # Input layer
    for i in range(num_layers):
        if i == 0:
            # First layer
            layers_list.append(layers.Dense(neurons, activation='relu', 
                                           kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))
        else:
            # Hidden layers
            layers_list.append(layers.Dense(neurons, activation='relu',
                                           kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))
        
        # Add dropout if specified
        if dropout_rate > 0:
            layers_list.append(layers.Dropout(dropout_rate))
    
    # Output layer
    layers_list.append(layers.Dense(1))
    
    # Create model
    model = keras.Sequential(layers_list)
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model with early stopping
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=0,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    return model, history

# Visualize the learning curves
def plot_history(histories, labels):
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

# Make predictions and visualize results
def plot_predictions(models, labels):
    # Generate points for smooth curve
    x_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the original data
    plt.scatter(x_train, y_train, alpha=0.4, label='Training data')
    plt.scatter(x_val, y_val, alpha=0.4, label='Validation data')
    
    # Plot predictions for each model
    for model, label in zip(models, labels):
        y_pred = model.predict(x_test)
        plt.plot(x_test, y_pred, linewidth=2, label=f'{label} prediction')
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    global x_train, y_train, x_val, y_val
    
    # 1. GENERATING SYNTHETIC DATA
    # Create a non-linear function with some noise
    x = np.linspace(-3, 3, 100)
    y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=len(x))

    # Split into training and validation sets
    indices = np.random.permutation(len(x))
    train_idx, val_idx = indices[:70], indices[70:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # Reshape for Keras
    x_train = x_train.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)

    # 2. VISUALIZE OUR DATA
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.7, label='Training data')
    plt.scatter(x_val, y_val, alpha=0.7, label='Validation data')
    plt.title('Synthetic Regression Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Training set size: {len(x_train)} samples")
    print(f"Validation set size: {len(x_val)} samples")

    # 3. EXPERIMENT WITH MODEL CAPACITY: UNDERFITTING VS OVERFITTING

    # 4. EXPERIMENT 1: UNDERFITTING (VERY SIMPLE MODEL)
    underfit_model, underfit_history = create_and_train_model(num_layers=1, neurons=2)

    # 5. EXPERIMENT 2: GOOD FIT
    good_model, good_history = create_and_train_model(num_layers=2, neurons=16)

    # 6. EXPERIMENT 3: OVERFITTING (VERY COMPLEX MODEL)
    overfit_model, overfit_history = create_and_train_model(num_layers=4, neurons=64)

    # 7. EXPERIMENT 4: REGULARIZED MODEL (L2)
    l2_model, l2_history = create_and_train_model(num_layers=4, neurons=64, l2_reg=0.01)

    # 8. EXPERIMENT 5: REGULARIZED MODEL (DROPOUT)
    dropout_model, dropout_history = create_and_train_model(num_layers=4, neurons=64, dropout_rate=0.3)

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

if __name__ == "__main__":
    main()
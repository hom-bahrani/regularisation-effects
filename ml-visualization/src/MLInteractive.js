import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';

// Make sure we're importing TensorFlow.js correctly
// Using require instead of import can sometimes help with module loading issues
const tf = require('@tensorflow/tfjs');

// Safely check TensorFlow.js availability
console.log("TensorFlow imported:", tf ? "Yes" : "No");
console.log("TensorFlow.js version:", tf && tf.version ? tf.version : "not found");

// Component styles
const styles = {
  container: {
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
    padding: '20px',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
  },
  controlPanel: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '20px',
    marginBottom: '30px',
    padding: '20px',
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    borderRadius: '8px',
  },
  controlGroup: {
    flex: '1 1 200px',
  },
  modelPerformance: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '10px',
    marginBottom: '20px',
  },
  metricCard: {
    flex: '1 1 150px',
    padding: '15px',
    borderRadius: '8px',
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    textAlign: 'center',
  },
  chartContainer: {
    marginBottom: '30px',
  },
  slider: {
    width: '100%',
  },
  button: {
    padding: '10px 15px',
    backgroundColor: '#4285F4',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    marginRight: '10px',
  },
  select: {
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    width: '100%',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    fontWeight: 'bold',
  },
  spinner: {
    display: 'inline-block',
    width: '20px',
    height: '20px',
    border: '3px solid rgba(0,0,0,.1)',
    borderRadius: '50%',
    borderTopColor: '#4285F4',
    animation: 'spin 1s ease-in-out infinite',
  },
};

// Function to generate synthetic data
const generateData = (numPoints, noise) => {
  // Generate x values between -3 and 3
  const x = Array.from({ length: numPoints }, (_, i) => 
    -3 + (i * 6 / (numPoints - 1))
  );
  
  // Generate y values with a quadratic function plus noise
  const y = x.map(val => 
    0.5 * Math.pow(val, 2) + val + 2 + (Math.random() - 0.5) * noise * 2
  );
  
  // Create array of data points
  const data = x.map((val, i) => ({
    x: val,
    y: y[i],
  }));
  
  // Shuffle and split data
  const shuffled = [...data].sort(() => 0.5 - Math.random());
  const splitIndex = Math.floor(shuffled.length * 0.7);
  
  const trainingData = shuffled.slice(0, splitIndex).map(point => ({
    ...point,
    type: 'training'
  }));
  
  const validationData = shuffled.slice(splitIndex).map(point => ({
    ...point,
    type: 'validation'
  }));
  
  return { trainingData, validationData, allData: [...trainingData, ...validationData] };
};

// Create a TensorFlow model based on parameters
const createModel = (layers, neurons, dropoutRate, l2Reg) => {
  try {
    // Check if TensorFlow is properly loaded
    if (!tf || typeof tf.sequential !== 'function') {
      throw new Error("TensorFlow.js is not properly loaded. Make sure the library is included in your dependencies.");
    }
    
    console.log("Creating a very simple model");
    
    // Create the simplest possible model
    const model = tf.sequential();
    
    // Add input layer (simplest possible configuration)
    model.add(tf.layers.dense({
      units: 1,
      inputShape: [1]
    }));
    
    // Compile with minimal options
    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError'
    });
    
    console.log("Basic model created successfully");
    return model;
  } catch (error) {
    console.error("DETAILED ERROR IN MODEL CREATION:", error);
    console.error("Error message:", error.message);
    console.error("Error stack:", error.stack);
    alert("TensorFlow.js error: " + error.message);
    throw error;
  }
};

// Main component
const MLInteractive = () => {
  // For tracking component mount state
  const isMounted = useRef(true);

  // State for data
  const [data, setData] = useState({ trainingData: [], validationData: [], allData: [] });
  
  // State for model parameters
  const [modelParams, setModelParams] = useState({
    layers: 2,
    neurons: 16,
    dropoutRate: 0,
    l2Reg: 0,
    epochs: 50,
    noiseLevel: 1,
    dataPoints: 48, // Reduced from 100 to match your screenshot
    batchSize: 16
  });

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMounted.current = false;
      // Clean up any tensors
      tf.disposeVariables();
    };
  }, []);
  
  // State for model performance metrics
  const [metrics, setMetrics] = useState({
    trainLoss: null,
    valLoss: null,
    trainMAE: null,
    valMAE: null
  });
  
  // State for training history
  const [history, setHistory] = useState([]);
  
  // State for predictions
  const [predictions, setPredictions] = useState([]);
  
  // State for available models
  const [models, setModels] = useState({
    underfit: null,
    goodFit: null,
    overfit: null,
    l2: null,
    dropout: null
  });
  
  // State for active model
  const [activeModel, setActiveModel] = useState('custom');
  
  // State for training process
  const [isTraining, setIsTraining] = useState(false);

  // Function to generate a smooth curve for display
  const generateCurve = useCallback((model) => {
    if (!model) return [];
    
    const testX = [];
    for (let i = 0; i < 200; i++) {
      testX.push(-4 + (i * 8 / 199));
    }
    
    // Convert to tensor
    const inputTensor = tf.tensor2d(testX, [testX.length, 1]);
    
    // Make predictions
    const outputTensor = model.predict(inputTensor);
    const outputData = outputTensor.dataSync();
    
    // Clean up tensors
    inputTensor.dispose();
    outputTensor.dispose();
    
    // Generate curve data
    return testX.map((x, i) => ({
      x,
      y: outputData[i],
    }));
  }, []);

  // Initialize data with error handling
  useEffect(() => {
    try {
      if (!modelParams) return;
      
      console.log("Generating data with parameters:", {
        dataPoints: modelParams.dataPoints,
        noiseLevel: modelParams.noiseLevel
      });
      
      const newData = generateData(
        modelParams.dataPoints || 48, 
        modelParams.noiseLevel || 1
      );
      
      console.log("Data generated:", {
        trainingSize: newData.trainingData.length,
        validationSize: newData.validationData.length
      });
      
      setData(newData);
    } catch (error) {
      console.error("Error generating data:", error);
      alert("Failed to generate data: " + error.message);
    }
  }, [modelParams?.dataPoints, modelParams?.noiseLevel]);
  
  // Function to train model
  const trainModel = async () => {
    console.log("Starting model training...");
    setIsTraining(true);
    const historyData = [];
    
    try {
      // Check if TensorFlow is available
      if (!tf) {
        throw new Error("TensorFlow.js is not loaded. Please check your dependencies.");
      }
      
      // Check if data is available
      if (!data || data.trainingData.length === 0 || data.validationData.length === 0) {
        throw new Error("No training or validation data available. Please regenerate data.");
      }
      
      // Create a very simple model (just one layer)
      const model = createModel();
      
      // Convert data to tensors
      console.log("Creating tensors from data");
      const trainX = tf.tensor2d(data.trainingData.map(d => [d.x]));
      const trainY = tf.tensor2d(data.trainingData.map(d => [d.y]));
      
      console.log("Starting training");
      // Just train the model without validation to keep it simple
      await model.fit(trainX, trainY, {
        epochs: 5, // Just a few epochs to test
        batchSize: 16,
        verbose: 1
      });
      
      console.log("Training completed successfully!");
      
      // Create a simple prediction curve
      const xs = tf.linspace(-4, 4, 100);
      const predictions = model.predict(xs.reshape([100, 1]));
      
      // Convert to array for display
      const predArray = Array.from(predictions.dataSync());
      const xsArray = Array.from(xs.dataSync());
      
      // Create prediction curve data
      const predictionCurve = xsArray.map((x, i) => ({
        x: x,
        y: predArray[i]
      }));
      
      // Update state
      setPredictions(predictionCurve);
      setHistory([{epoch: 0, trainLoss: 0}]); // Dummy history
      
      // Clean up tensors
      trainX.dispose();
      trainY.dispose();
      xs.dispose();
      predictions.dispose();
      
    } catch (error) {
      console.error("Training error:", error);
      alert("Training failed: " + error.message);
    } finally {
      setIsTraining(false);
    }
  };
  
  // Function to train all preset models
  const trainAllModels = async () => {
    setIsTraining(true);
    const newModels = {};
    
    try {
      // Underfit model (1 layer, 2 neurons)
      const underfit = createModel(1, 2, 0, 0);
      await trainSingleModel(underfit, 'underfit');
      newModels.underfit = underfit;
      
      // Good fit model (2 layers, 16 neurons)
      const goodFit = createModel(2, 16, 0, 0);
      await trainSingleModel(goodFit, 'goodFit');
      newModels.goodFit = goodFit;
      
      // Overfit model (4 layers, 64 neurons)
      const overfit = createModel(4, 64, 0, 0);
      await trainSingleModel(overfit, 'overfit');
      newModels.overfit = overfit;
      
      // L2 regularized model (4 layers, 64 neurons, L2 reg)
      const l2 = createModel(4, 64, 0, 0.01);
      await trainSingleModel(l2, 'l2');
      newModels.l2 = l2;
      
      // Dropout model (4 layers, 64 neurons, dropout)
      const dropout = createModel(4, 64, 0.3, 0);
      await trainSingleModel(dropout, 'dropout');
      newModels.dropout = dropout;
      
      // Update models
      setModels(newModels);
      
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      setIsTraining(false);
    }
  };
  
  // Helper function to train an individual model
  const trainSingleModel = async (model, modelName) => {
    // Prepare training data
    const trainX = tf.tensor2d(data.trainingData.map(d => [d.x]));
    const trainY = tf.tensor2d(data.trainingData.map(d => [d.y]));
    
    // Prepare validation data
    const valX = tf.tensor2d(data.validationData.map(d => [d.x]));
    const valY = tf.tensor2d(data.validationData.map(d => [d.y]));
    
    // Train the model
    await model.fit(trainX, trainY, {
      epochs: 50,
      batchSize: 16,
      validationData: [valX, valY],
    });
    
    // Clean up tensors
    trainX.dispose();
    trainY.dispose();
    valX.dispose();
    valY.dispose();
  };
  
  // Function to handle parameter changes
  const handleParamChange = (param, value) => {
    setModelParams({
      ...modelParams,
      [param]: value
    });
  };
  
  // Function to regenerate data
  const handleRegenerateData = () => {
    const newData = generateData(modelParams.dataPoints, modelParams.noiseLevel);
    setData(newData);
    setPredictions([]);
    setHistory([]);
  };
  
  // Get active model predictions
  const getActiveModelPredictions = useCallback(() => {
    if (activeModel === 'custom') {
      return predictions;
    } else if (models[activeModel]) {
      return generateCurve(models[activeModel]);
    }
    return [];
  }, [activeModel, models, predictions, generateCurve]);
  
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>Interactive Machine Learning Visualization</h1>
        <p>Explore underfitting, overfitting, and regularization effects in real-time</p>
      </div>
      
      <div style={styles.controlPanel}>
        <div style={styles.controlGroup}>
          <h3>Data Generation</h3>
          <div>
            <label style={styles.label}>Number of data points: {modelParams.dataPoints}</label>
            <input 
              type="range" 
              min="20" 
              max="500"
              value={modelParams.dataPoints}
              onChange={(e) => handleParamChange('dataPoints', parseInt(e.target.value))}
              style={styles.slider}
            />
          </div>
          <div>
            <label style={styles.label}>Noise level: {modelParams.noiseLevel.toFixed(1)}</label>
            <input 
              type="range" 
              min="0" 
              max="3" 
              step="0.1"
              value={modelParams.noiseLevel}
              onChange={(e) => handleParamChange('noiseLevel', parseFloat(e.target.value))}
              style={styles.slider}
            />
          </div>
          <button 
            style={styles.button} 
            onClick={handleRegenerateData}
            disabled={isTraining}
          >
            Regenerate Data
          </button>
        </div>
        
        <div style={styles.controlGroup}>
          <h3>Model Architecture</h3>
          <div>
            <label style={styles.label}>Number of hidden layers: {modelParams.layers}</label>
            <input 
              type="range" 
              min="1" 
              max="5"
              value={modelParams.layers}
              onChange={(e) => handleParamChange('layers', parseInt(e.target.value))}
              style={styles.slider}
              disabled={isTraining}
            />
          </div>
          <div>
            <label style={styles.label}>Neurons per layer: {modelParams.neurons}</label>
            <input 
              type="range" 
              min="1" 
              max="128"
              value={modelParams.neurons}
              onChange={(e) => handleParamChange('neurons', parseInt(e.target.value))}
              style={styles.slider}
              disabled={isTraining}
            />
          </div>
        </div>
        
        <div style={styles.controlGroup}>
          <h3>Regularization</h3>
          <div>
            <label style={styles.label}>Dropout rate: {modelParams.dropoutRate.toFixed(2)}</label>
            <input 
              type="range" 
              min="0" 
              max="0.5" 
              step="0.05"
              value={modelParams.dropoutRate}
              onChange={(e) => handleParamChange('dropoutRate', parseFloat(e.target.value))}
              style={styles.slider}
              disabled={isTraining}
            />
          </div>
          <div>
            <label style={styles.label}>L2 regularization: {modelParams.l2Reg.toFixed(3)}</label>
            <input 
              type="range" 
              min="0" 
              max="0.05" 
              step="0.001"
              value={modelParams.l2Reg}
              onChange={(e) => handleParamChange('l2Reg', parseFloat(e.target.value))}
              style={styles.slider}
              disabled={isTraining}
            />
          </div>
        </div>
        
        <div style={styles.controlGroup}>
          <h3>Training</h3>
          <div>
            <label style={styles.label}>Epochs: {modelParams.epochs}</label>
            <input 
              type="range" 
              min="10" 
              max="100"
              value={modelParams.epochs}
              onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
              style={styles.slider}
              disabled={isTraining}
            />
          </div>
          <div>
            <label style={styles.label}>Batch size: {modelParams.batchSize}</label>
            <input 
              type="range" 
              min="1" 
              max="64"
              value={modelParams.batchSize}
              onChange={(e) => handleParamChange('batchSize', parseInt(e.target.value))}
              style={styles.slider}
              disabled={isTraining}
            />
          </div>
          <div>
            <button 
              style={{...styles.button, marginTop: '10px'}} 
              onClick={trainModel}
              disabled={isTraining}
            >
              {isTraining ? 
                <><span style={styles.spinner}></span> Training...</> : 
                'Train Model'}
            </button>
          </div>
        </div>
        
        <div style={styles.controlGroup}>
          <h3>Preset Models</h3>
          <div>
            <label style={styles.label}>Select Model</label>
            <select 
              style={styles.select} 
              value={activeModel}
              onChange={(e) => setActiveModel(e.target.value)}
              disabled={isTraining}
            >
              <option value="custom">Custom Model</option>
              <option value="underfit" disabled={!models.underfit}>Underfit</option>
              <option value="goodFit" disabled={!models.goodFit}>Good Fit</option>
              <option value="overfit" disabled={!models.overfit}>Overfit</option>
              <option value="l2" disabled={!models.l2}>L2 Regularized</option>
              <option value="dropout" disabled={!models.dropout}>Dropout</option>
            </select>
          </div>
          <button 
            style={{...styles.button, marginTop: '10px'}} 
            onClick={trainAllModels}
            disabled={isTraining}
          >
            {isTraining ? 
              <><span style={styles.spinner}></span> Training All Models...</> : 
              'Train All Preset Models'}
          </button>
        </div>
      </div>
      
      {metrics.trainLoss !== null && (
        <div style={styles.modelPerformance}>
          <div style={styles.metricCard}>
            <h4>Training Loss</h4>
            <p>{metrics.trainLoss.toFixed(4)}</p>
          </div>
          <div style={styles.metricCard}>
            <h4>Validation Loss</h4>
            <p>{metrics.valLoss.toFixed(4)}</p>
          </div>
          <div style={styles.metricCard}>
            <h4>Training MAE</h4>
            <p>{metrics.trainMAE.toFixed(4)}</p>
          </div>
          <div style={styles.metricCard}>
            <h4>Validation MAE</h4>
            <p>{metrics.valMAE.toFixed(4)}</p>
          </div>
        </div>
      )}
      
      <div style={styles.chartContainer}>
        <h3>Data and Model Predictions</h3>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="x" />
            <YAxis type="number" dataKey="y" name="y" />
            <ZAxis type="number" range={[60]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            
            {/* Training data points */}
            <Scatter 
              name="Training Data" 
              data={data.trainingData} 
              fill="#8884d8" 
              shape="circle"
            />
            
            {/* Validation data points */}
            <Scatter 
              name="Validation Data" 
              data={data.validationData} 
              fill="#82ca9d" 
              shape="circle"
            />
            
            {/* Model predictions */}
            <Scatter
              name={`${activeModel === 'custom' ? 'Custom' : activeModel} Model Predictions`}
              data={getActiveModelPredictions()}
              fill="#ff7300"
              line
              shape="none"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      
      {history.length > 0 && (
        <div style={styles.chartContainer}>
          <h3>Training History</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={history} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epochs', position: 'insideBottom', offset: -10 }} />
              <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke="#8884d8" activeDot={{ r: 8 }} />
              <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default MLInteractive;
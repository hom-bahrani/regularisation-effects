import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
// Material UI imports
import { 
  Container, Box, Typography, Grid, Slider, Button, 
  FormControl, InputLabel, Select, MenuItem,
  Paper, Card, CardContent, CardHeader, CircularProgress
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Import TensorFlow.js
const tf = require('@tensorflow/tfjs');

// Create Material UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#4285F4',
    },
    secondary: {
      main: '#34A853',
    },
    error: {
      main: '#EA4335',
    },
    warning: {
      main: '#FBBC05',
    },
    background: {
      default: '#f5f5f5',
    }
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 500,
    },
    h5: {
      fontWeight: 500,
    }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

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
    
    console.log("Creating model with parameters:", {
      layers: layers,
      neurons: neurons,
      dropoutRate: dropoutRate,
      l2Reg: l2Reg
    });
    
    // Input validation
    if (!layers || layers < 1) layers = 1;
    if (!neurons || neurons < 1) neurons = 16;
    if (!dropoutRate) dropoutRate = 0;
    if (!l2Reg) l2Reg = 0;
    
    const model = tf.sequential();
    
    // First layer needs input shape
    model.add(tf.layers.dense({
      units: neurons, 
      activation: 'relu',
      inputShape: [1],
      kernelRegularizer: l2Reg > 0 ? tf.regularizers.l2({ l2: l2Reg }) : null
    }));
    
    // Add remaining layers
    for (let i = 1; i < layers; i++) {
      model.add(tf.layers.dense({
        units: neurons,
        activation: 'relu',
        kernelRegularizer: l2Reg > 0 ? tf.regularizers.l2({ l2: l2Reg }) : null
      }));
      
      // Add dropout if specified
      if (dropoutRate > 0) {
        model.add(tf.layers.dropout({ rate: dropoutRate }));
      }
    }
    
    // Output layer
    model.add(tf.layers.dense({ units: 1 }));
    
    // Compile model
    model.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError',
      metrics: ['mae']
    });
    
    console.log("Model created successfully");
    model.summary();
    return model;
  } catch (error) {
    console.error("Error creating model:", error);
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
    dataPoints: 48,
    batchSize: 16
  });
  
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMounted.current = false;
      // Clean up any tensors
      tf.disposeVariables();
    };
  }, []);

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
      
      // Create model based on current parameters
      const model = createModel(
        modelParams.layers,
        modelParams.neurons,
        modelParams.dropoutRate,
        modelParams.l2Reg
      );
      
      // Convert data to tensors
      console.log("Creating tensors from data");
      const trainX = tf.tensor2d(data.trainingData.map(d => [d.x]));
      const trainY = tf.tensor2d(data.trainingData.map(d => [d.y]));
      const valX = tf.tensor2d(data.validationData.map(d => [d.x]));
      const valY = tf.tensor2d(data.validationData.map(d => [d.y]));
      
      console.log("Starting training");
      await model.fit(trainX, trainY, {
        epochs: modelParams.epochs,
        batchSize: modelParams.batchSize,
        validationData: [valX, valY],
        verbose: 1,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            historyData.push({
              epoch,
              trainLoss: logs.loss,
              valLoss: logs.val_loss,
              trainMAE: logs.mae,
              valMAE: logs.val_mae
            });
            
            // Update history periodically
            if (epoch % 5 === 0 || epoch === modelParams.epochs - 1) {
              setHistory([...historyData]);
            }
          }
        }
      });
      
      console.log("Training completed");
      
      // Evaluate the model
      const trainEval = model.evaluate(trainX, trainY);
      const valEval = model.evaluate(valX, valY);
      
      // Update metrics
      setMetrics({
        trainLoss: trainEval[0].dataSync()[0],
        trainMAE: trainEval[1].dataSync()[0],
        valLoss: valEval[0].dataSync()[0],
        valMAE: valEval[1].dataSync()[0]
      });
      
      // Generate predictions
      const predictedCurve = generateCurve(model);
      setPredictions(predictedCurve);
      
      // Clean up tensors
      trainX.dispose();
      trainY.dispose();
      valX.dispose();
      valY.dispose();
      trainEval[0].dispose();
      trainEval[1].dispose();
      valEval[0].dispose();
      valEval[1].dispose();
      
      // Update history
      setHistory(historyData);
      
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
      alert("Failed to train preset models: " + error.message);
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
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Interactive Machine Learning Visualization
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Explore underfitting, overfitting, and regularization effects in real-time
          </Typography>
        </Box>
        
        <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
          <Grid container spacing={3}>
            {/* Data Generation */}
            <Grid item xs={12} md={4} lg={2.4}>
              <Typography variant="h6" gutterBottom>
                Data Generation
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography id="data-points-slider" gutterBottom>
                  Number of data points: {modelParams.dataPoints}
                </Typography>
                <Slider
                  aria-labelledby="data-points-slider"
                  min={20}
                  max={500}
                  value={modelParams.dataPoints}
                  onChange={(_, value) => handleParamChange('dataPoints', value)}
                  disabled={isTraining}
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography id="noise-level-slider" gutterBottom>
                  Noise level: {modelParams.noiseLevel.toFixed(1)}
                </Typography>
                <Slider
                  aria-labelledby="noise-level-slider"
                  min={0}
                  max={3}
                  step={0.1}
                  value={modelParams.noiseLevel}
                  onChange={(_, value) => handleParamChange('noiseLevel', value)}
                  disabled={isTraining}
                />
              </Box>
              <Button 
                variant="contained" 
                fullWidth
                onClick={handleRegenerateData}
                disabled={isTraining}
              >
                Regenerate Data
              </Button>
            </Grid>
            
            {/* Model Architecture */}
            <Grid item xs={12} md={4} lg={2.4}>
              <Typography variant="h6" gutterBottom>
                Model Architecture
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography id="layers-slider" gutterBottom>
                  Number of hidden layers: {modelParams.layers}
                </Typography>
                <Slider
                  aria-labelledby="layers-slider"
                  min={1}
                  max={5}
                  value={modelParams.layers}
                  onChange={(_, value) => handleParamChange('layers', value)}
                  disabled={isTraining}
                  marks
                  step={1}
                  valueLabelDisplay="auto"
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography id="neurons-slider" gutterBottom>
                  Neurons per layer: {modelParams.neurons}
                </Typography>
                <Slider
                  aria-labelledby="neurons-slider"
                  min={1}
                  max={128}
                  value={modelParams.neurons}
                  onChange={(_, value) => handleParamChange('neurons', value)}
                  disabled={isTraining}
                />
              </Box>
            </Grid>
            
            {/* Regularization */}
            <Grid item xs={12} md={4} lg={2.4}>
              <Typography variant="h6" gutterBottom>
                Regularization
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography id="dropout-slider" gutterBottom>
                  Dropout rate: {modelParams.dropoutRate.toFixed(2)}
                </Typography>
                <Slider
                  aria-labelledby="dropout-slider"
                  min={0}
                  max={0.5}
                  step={0.05}
                  value={modelParams.dropoutRate}
                  onChange={(_, value) => handleParamChange('dropoutRate', value)}
                  disabled={isTraining}
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography id="l2-slider" gutterBottom>
                  L2 regularization: {modelParams.l2Reg.toFixed(3)}
                </Typography>
                <Slider
                  aria-labelledby="l2-slider"
                  min={0}
                  max={0.05}
                  step={0.001}
                  value={modelParams.l2Reg}
                  onChange={(_, value) => handleParamChange('l2Reg', value)}
                  disabled={isTraining}
                />
              </Box>
            </Grid>
            
            {/* Training */}
            <Grid item xs={12} md={6} lg={2.4}>
              <Typography variant="h6" gutterBottom>
                Training
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography id="epochs-slider" gutterBottom>
                  Epochs: {modelParams.epochs}
                </Typography>
                <Slider
                  aria-labelledby="epochs-slider"
                  min={10}
                  max={100}
                  value={modelParams.epochs}
                  onChange={(_, value) => handleParamChange('epochs', value)}
                  disabled={isTraining}
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography id="batch-size-slider" gutterBottom>
                  Batch size: {modelParams.batchSize}
                </Typography>
                <Slider
                  aria-labelledby="batch-size-slider"
                  min={1}
                  max={64}
                  value={modelParams.batchSize}
                  onChange={(_, value) => handleParamChange('batchSize', value)}
                  disabled={isTraining}
                />
              </Box>
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={trainModel}
                disabled={isTraining}
                startIcon={isTraining && <CircularProgress size={20} color="inherit" />}
              >
                {isTraining ? 'Training...' : 'Train Model'}
              </Button>
            </Grid>
            
            {/* Preset Models */}
            <Grid item xs={12} md={6} lg={2.4}>
              <Typography variant="h6" gutterBottom>
                Preset Models
              </Typography>
              <Box sx={{ mb: 2 }}>
                <FormControl fullWidth>
                  <InputLabel id="model-select-label">Select Model</InputLabel>
                  <Select
                    labelId="model-select-label"
                    value={activeModel}
                    label="Select Model"
                    onChange={(e) => setActiveModel(e.target.value)}
                    disabled={isTraining}
                  >
                    <MenuItem value="custom">Custom Model</MenuItem>
                    <MenuItem value="underfit" disabled={!models.underfit}>Underfit</MenuItem>
                    <MenuItem value="goodFit" disabled={!models.goodFit}>Good Fit</MenuItem>
                    <MenuItem value="overfit" disabled={!models.overfit}>Overfit</MenuItem>
                    <MenuItem value="l2" disabled={!models.l2}>L2 Regularized</MenuItem>
                    <MenuItem value="dropout" disabled={!models.dropout}>Dropout</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <Button
                variant="contained"
                color="secondary"
                fullWidth
                onClick={trainAllModels}
                disabled={isTraining}
                startIcon={isTraining && <CircularProgress size={20} color="inherit" />}
              >
                {isTraining ? 'Training All Models...' : 'Train All Preset Models'}
              </Button>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Metrics Cards */}
        {metrics.trainLoss !== null && (
          <Grid container spacing={2} sx={{ mb: 4 }}>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Training Loss
                  </Typography>
                  <Typography variant="h5">
                    {metrics.trainLoss.toFixed(4)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Validation Loss
                  </Typography>
                  <Typography variant="h5">
                    {metrics.valLoss.toFixed(4)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Training MAE
                  </Typography>
                  <Typography variant="h5">
                    {metrics.trainMAE.toFixed(4)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    Validation MAE
                  </Typography>
                  <Typography variant="h5">
                    {metrics.valMAE.toFixed(4)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
        
        {/* Data and Predictions Chart */}
        <Card sx={{ mb: 4 }}>
          <CardHeader title="Data and Model Predictions" />
          <CardContent>
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
                  fillOpacity={0.7}
                />
                
                {/* Validation data points */}
                <Scatter 
                  name="Validation Data" 
                  data={data.validationData} 
                  fill="#82ca9d" 
                  shape="circle"
                  fillOpacity={0.7}
                />
                
                {/* Model predictions */}
                <Scatter
                  name={`${activeModel === 'custom' ? 'Custom' : activeModel} Model Predictions`}
                  data={getActiveModelPredictions()}
                  fill="#ff7300"
                  line={{ stroke: '#ff7300', strokeWidth: 2 }}
                  shape="none"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* Training History Chart */}
        {history.length > 0 && (
          <Card>
            <CardHeader title="Training History" />
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={history} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ value: 'Epochs', position: 'insideBottom', offset: -10 }} 
                  />
                  <YAxis 
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} 
                  />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="trainLoss" 
                    name="Training Loss" 
                    stroke="#8884d8" 
                    activeDot={{ r: 8 }}
                    strokeWidth={2} 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="valLoss" 
                    name="Validation Loss" 
                    stroke="#82ca9d"
                    strokeWidth={2} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </Container>
    </ThemeProvider>
  );
};

export default MLInteractive;
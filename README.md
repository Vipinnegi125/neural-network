# Neural Network Project: ETF Price Movement Prediction

## Project Overview

This project implements advanced deep learning models to predict directional price movements (up/down) for multiple Exchange-Traded Funds (ETFs) using historical daily returns data. The analysis spans from 2018 to 2022 and includes comprehensive data analysis, statistical testing, and two distinct neural network architectures.

## Objectives

- **Data Analysis**: Analyze correlations and performance metrics across five major ETFs
- **Stationarity Testing**: Verify time-series properties using statistical tests
- **Model Development**: Build and evaluate two different neural network architectures
- **Performance Evaluation**: Compare model accuracy on training and out-of-sample test data

## Project Structure

```
neural-network/
├── Group_project_Neural_network.ipynb    # Main Jupyter Notebook with all analysis and models
├── Group_project_Neural_network.html     # HTML export of the notebook
├── README.md                              # This file
```

## Dataset

**Source**: Yahoo Finance (yfinance)  
**Period**: January 1, 2018 - December 30, 2022  
**Assets Analyzed** (5 ETFs):
- **SPY**: S&P 500 (U.S. Equities)
- **TLT**: 20+ Year Treasury Bonds
- **SHY**: 1-3 Year Treasury Bonds
- **GLD**: Gold Commodity
- **DBO**: Commodities Basket

## Methodology

### Phase 1: Exploratory Data Analysis

1. **Data Preparation**
   - Download historical closing prices for all 5 ETFs
   - Calculate daily log returns and percentage returns
   - Handle missing values

2. **Statistical Analysis**
   - Correlation heatmap showing relationships between assets
   - Normalized performance visualization (base 100)
   - Daily log returns analysis

3. **Stationarity Testing**
   - **Augmented Dickey-Fuller (ADF) Test**: Tests for unit roots
   - **KPSS Test**: Tests for mean/trend stationarity
   - Validates time-series properties for modeling

### Phase 2: Deep Learning Models

#### Model 1: GAF-CNN (Gramian Angular Field - Convolutional Neural Network)

**Architecture**:
- Input: 20-day lookback window transformed into 2D Gramian Angular Field images
- Layer 1: Conv2D (32 filters, 3×3 kernel) → MaxPooling(2×2)
- Layer 2: Conv2D (64 filters, 3×3 kernel) → MaxPooling(2×2)
- Flatten → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)

**Features**:
- Converts 1D return sequences into 2D image representations
- Captures temporal patterns as visual motifs
- Binary classification: Up (1) vs Down (0) over 25-day forward horizon
- Trained with class weights to handle imbalanced data

**Evaluation Metrics**:
- In-sample training accuracy
- Out-of-sample test accuracy
- Confusion matrices for each ETF
- Training/validation accuracy and loss curves

#### Model 2: MLP (Multi-Layer Perceptron)

**Architecture**:
- Input: 5 features representing different lookback windows (5, 10, 20, 60, 90 days)
- Layer 1: Dense(25, ReLU) → Dropout(0.2)
- Layer 2: Dense(15, ReLU) → Dropout(0.2)
- Layer 3: Dense(10, ReLU) → Dropout(0.2)
- Output: Dense(1, Sigmoid)

**Features**:
- Multi-horizon feature engineering (5 different historical windows)
- Standardized input features (StandardScaler)
- Binary classification of 25-day forward returns
- Early stopping to prevent overfitting

**Evaluation Metrics**:
- In-sample and out-of-sample accuracy
- Confusion matrices and performance summaries

## Key Findings

- Both models are trained on historical data with a 20% test split (chronological)
- Class weighting is applied to account for natural asset drift bias
- Early stopping and dropout regularization prevent overfitting
- Performance is evaluated on out-of-sample (test) data for realistic assessment

## Technical Requirements

### Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `yfinance`: Yahoo Finance data download
- `tensorflow` / `keras`: Deep learning models
- `scikit-learn`: Preprocessing, metrics, and evaluation
- `matplotlib` & `seaborn`: Data visualization
- `statsmodels`: Statistical testing (ADF, KPSS)
- `pyts`: Gramian Angular Field transformations

### Installation

```bash
pip install numpy pandas yfinance tensorflow scikit-learn matplotlib seaborn statsmodels pyts
```

## Model Training & Evaluation

### Data Split Strategy
- **Training**: 80% of historical data (chronological)
- **Testing**: 20% of most recent data (out-of-sample)
- **No shuffling**: Maintains temporal integrity of time-series data

### Hyperparameters

**Both Models**:
- Optimizer: Adam (learning rate: 1e-4 for CNN, 1e-5 for MLP)
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Epochs: 40 (with Early Stopping, patience=10)

**CNN-Specific**:
- Lookback window: 20 days
- Forward horizon: 25 days

**MLP-Specific**:
- Feature lookback windows: [5, 10, 20, 60, 90] days
- Forward horizon: 25 days
- Dropout rate: 0.2

## Results

The notebook generates comprehensive evaluation reports including:

1. **Correlation Analysis**: Heatmap showing asset relationships
2. **Normalized Performance Charts**: Indexed returns from 2018-2022
3. **GAF Visualizations**: Sample 2D image representations of return patterns
4. **Training Curves**: Accuracy and loss over epochs
5. **Confusion Matrices**: Classification performance for each ETF
6. **Summary Tables**: In-sample vs out-of-sample accuracy comparison

## Usage

1. Open `Group_project_Neural_network.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially (Steps 1-3 for analysis, subsequent cells for models)
3. Models automatically train on downloaded data and generate visualizations
4. Review accuracy metrics and confusion matrices in console output

## Output Files

- **Visualizations**: 
  - Correlation heatmaps
  - Normalized performance plots
  - Log returns charts
  - GAF-CNN diagnostic plots (4-subplot grid per ETF)
  - Training curves and confusion matrices

- **Console Output**:
  - Stationarity test results
  - In-sample and out-of-sample accuracy scores
  - Model performance summary table

## Future Improvements

- Ensemble methods combining CNN and MLP predictions
- LSTM/GRU architectures for sequence modeling
- Additional technical indicators as features
- Walk-forward validation for more robust backtesting
- Transaction cost and slippage analysis
- Risk-adjusted performance metrics (Sharpe ratio, Sortino ratio)

## Notes

- All models are **directional predictors** (up/down), not absolute price forecasters
- 25-day forward horizon chosen to balance signal-to-noise ratio
- Results are **NOT investment advice** and should not be used for live trading
- Historical performance does not guarantee future results
- Class imbalance in training data is handled through weighted loss functions

## Author

Group Project - Neural Network Analysis Team

## License

This project is provided for educational and research purposes.
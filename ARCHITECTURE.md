# Aircraft Engine Health Prediction - Architecture Documentation

## Overview

This document provides a comprehensive architectural overview of the Aircraft Engine Health Prediction system. The system implements a machine learning pipeline for predicting Remaining Useful Life (RUL) of aircraft engines using sensor data from the NASA C-MAPSS dataset.

## System Architecture

### High-Level Components

1. **Data Layer**: Raw data storage and processed artifacts
2. **ML Pipeline**: Data ingestion, transformation, and model training
3. **Prediction Service**: Real-time inference pipeline
4. **Web Interface**: Flask-based user interface for predictions
5. **Logging and Exception Handling**: Centralized error management

### Data Flow

```
Raw Data (NASA C-MAPSS)
    ↓
Data Ingestion
    ↓
Data Transformation
    ↓
Model Training
    ↓
Trained Model + Preprocessor
    ↓
Prediction Pipeline
    ↓
Web Interface
    ↓
User Predictions
```

## Detailed Component Breakdown

### 1. Data Layer

#### Raw Data Structure
- **Location**: `data/raw/`
- **Format**: Space-separated text files
- **Datasets**: FD001 (used), FD002-FD004 (available)
- **Columns**:
  - `unit_nr`: Engine unit identifier
  - `time_cycles`: Operational cycles
  - `setting_1-3`: Operational settings
  - `s_1-21`: Sensor measurements
- **Target**: Remaining Useful Life (RUL) calculated as max_cycles - current_cycles

#### Processed Data
- **Location**: `artifacts/`
- **Files**:
  - `train.csv`: Training data with RUL
  - `test.csv`: Test data
  - `data.csv`: Combined dataset
  - `model.pkl`: Trained ML model
  - `preprocessor.pkl`: Data preprocessing pipeline

### 2. ML Pipeline Components

#### Data Ingestion (`src/components/data_ingestion.py`)
- **Purpose**: Load raw data and prepare for training
- **Inputs**: Raw text files from `data/raw/`
- **Outputs**: CSV files in `artifacts/`
- **Key Operations**:
  - Parse space-separated values
  - Calculate RUL for training data
  - Split into train/test sets

#### Data Transformation (`src/components/data_transformation.py`)
- **Purpose**: Feature engineering and preprocessing
- **Inputs**: Raw CSV data
- **Outputs**: Scaled arrays + preprocessor object
- **Key Operations**:
  - Feature selection (10 sensors used)
  - Train/test split
  - Data scaling/normalization
  - Save preprocessor for prediction

#### Model Trainer (`src/components/model_trainer.py`)
- **Purpose**: Train and evaluate ML models
- **Inputs**: Preprocessed arrays
- **Outputs**: Trained model saved as pickle
- **Key Operations**:
  - Model selection and hyperparameter tuning
  - Cross-validation
  - Performance evaluation
  - Model serialization

### 3. Prediction Pipeline

#### Predict Pipeline (`src/pipelines/predict_pipeline.py`)
- **Purpose**: Real-time RUL prediction
- **Inputs**: User sensor data (10 features)
- **Outputs**: RUL prediction in cycles
- **Key Operations**:
  - Load trained model and preprocessor
  - Transform input data
  - Make prediction
  - Return RUL value

#### Custom Data Class
- **Purpose**: Structure user input data
- **Features**: s_2, s_3, s_4, s_7, s_11, s_12, s_15, s_17, s_20, s_21
- **Output**: Pandas DataFrame for pipeline

### 4. Web Application

#### Flask App (`app.py`)
- **Framework**: Flask 3.1.2
- **Routes**:
  - `/`: Home page
  - `/project`: Project information
  - `/contact`: Contact page
  - `/predictdata`: Prediction endpoint (GET/POST)
- **Health Classification Logic**:
  - RUL > 150: HEALTHY (green)
  - 50 < RUL ≤ 150: DEGRADATION (warning)
  - RUL ≤ 50: FAILURE IMMINENT (danger)

#### Templates
- **Location**: `templates/`
- **Files**: base.html, home.html, project.html, contact.html
- **Features**: Bootstrap styling, dynamic results display

### 5. Utilities and Infrastructure

#### Exception Handling (`src/exception.py`)
- Custom exception class for error management
- Centralized error logging and handling

#### Logging (`src/logger.py`)
- Configurable logging system
- Logs pipeline operations and errors

#### Utils (`src/utils.py`)
- Helper functions for model loading/saving
- Common data operations

## Technology Stack

### Core Dependencies
- **Python**: 3.8+ (primary language)
- **Flask**: 3.1.2 (web framework)
- **Scikit-learn**: ML algorithms and preprocessing
- **Pandas**: 2.0+ (data manipulation)
- **NumPy**: 1.24+ (numerical operations)

### Development Tools
- **Jupyter**: Experimentation and EDA
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization

### Environment
- **Virtual Environment**: `aircraft_venv/`
- **Package Management**: pip with requirements.txt
- **Setup**: setup.py for package installation

## Data Processing Pipeline

### Training Phase
1. Load raw FD001 training data
2. Calculate RUL for each cycle
3. Select relevant sensor features
4. Split data (80/20 train/test)
5. Apply preprocessing (scaling)
6. Train regression model
7. Evaluate performance
8. Save model and preprocessor

### Prediction Phase
1. Receive user sensor inputs
2. Load preprocessor and model
3. Transform input data
4. Make RUL prediction
5. Classify health status
6. Return results to user

## Model Details

### Current Implementation
- **Algorithm**: Likely Random Forest or Gradient Boosting (based on typical RUL tasks)
- **Features**: 10 selected sensors (s_2, s_3, s_4, s_7, s_11, s_12, s_15, s_17, s_20, s_21)
- **Target**: RUL in operational cycles
- **Evaluation**: RMSE, MAE, R² score

### Feature Selection Rationale
- Sensors chosen based on correlation with RUL
- Excludes constant or irrelevant sensors
- Balances predictive power with computational efficiency

## Deployment Considerations

### Local Development
- Run `python app.py` for web interface
- Access at http://localhost:5000
- Debug mode enabled for development

### Production Deployment
- Disable debug mode
- Use production WSGI server (gunicorn)
- Configure proper logging
- Set up monitoring and alerts

## Future Enhancements

### Potential Improvements
- Multiple dataset support (FD002-FD004)
- Advanced ML models (LSTM, CNN for time series)
- Real-time streaming data integration
- API endpoints for external systems
- Model versioning and A/B testing
- Containerization with Docker

### Scalability Considerations
- Batch prediction capabilities
- Model serving optimization
- Database integration for historical data
- Cloud deployment options

## Security and Best Practices

### Data Security
- No sensitive data in repository
- Virtual environment isolation
- Input validation in web forms

### Code Quality
- Modular component design
- Exception handling throughout
- Logging for debugging and monitoring
- Type hints and documentation

This architecture provides a solid foundation for aircraft engine health monitoring with room for expansion and improvement.
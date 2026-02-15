# Aircraft Engine Health Prediction

A machine learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using sensor data. This project implements an end-to-end ML pipeline with data processing, model training, and a web-based prediction interface.
## LIVE: https://aegis-aero.onrender.com
## Features

- **Data Processing**: Automated ingestion and preprocessing of NASA C-MAPSS dataset
- **ML Pipeline**: Feature engineering, model training, and evaluation
- **Web Application**: Flask-based interface for real-time RUL predictions
- **Health Monitoring**: Categorizes engine health as Healthy, Degrading, or Failure Imminent
- **Modular Architecture**: Organized components for data, training, and prediction pipelines

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aircraft-engine-ml.git
   cd aircraft-engine-ml
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv aircraft_venv
   aircraft_venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the training pipeline:
   ```bash
   python src/pipelines/train_pipeline.py
   ```

5. Start the web application:
   ```bash
   python app.py
   ```

## Usage

### Web Interface
1. Open http://localhost:5000 in your browser
2. Navigate to the prediction page
3. Enter sensor measurements (s_2, s_3, s_4, s_7, s_11, s_12, s_15, s_17, s_20, s_21)
4. Get RUL prediction and health status

### API Usage
The Flask app provides endpoints for programmatic access to predictions.

## Data

This project uses the NASA C-MAPSS dataset (FD001 subset), which contains:
- **100 training trajectories** from engines running to failure
- **100 test trajectories** ending before failure
- **21 sensor measurements** per cycle
- **3 operational settings**
- **Target**: Remaining Useful Life (RUL) in cycles

Data includes sensor noise and manufacturing variations. The goal is to predict how many operational cycles remain before engine failure.

## Technologies

- **Python** 3.8+
- **Flask** for web framework
- **Scikit-learn** for ML algorithms
- **Pandas/Numpy** for data manipulation
- **Matplotlib/Seaborn** for visualization
- **Jupyter** for experimentation

## Project Structure

```
├── app.py                 # Flask web application
├── src/
│   ├── components/        # ML pipeline components
│   ├── pipelines/         # Training and prediction pipelines
│   └── utils/             # Utility functions
├── data/raw/              # Raw NASA dataset
├── artifacts/             # Processed data and trained models
├── notebooks/             # EDA and experimentation
├── templates/             # HTML templates
└── static/                # CSS/JS assets
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

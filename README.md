# 💗Heart Disease Prediction App

This project is a machine learning web application for predicting the likelihood of heart disease based on user input data. It uses a trained model and data preprocessing pipeline to provide predictions and visualizations.

## Features

- Predicts heart disease risk using user-provided health data
- Utilizes a trained LightGBM model (`lgb.pkl`)
- Data normalization with MinMaxScaler (`minmax_scaler.joblib`)
- Visualizations: ROC curve, confusion matrix, classification results

## Files

- `app.py`: Main application scripts
- `lgb.pkl`: Trained LightGBM model
- `minmax_scaler.joblib`: Data scaler for preprocessing
- `requirements.txt`: List of required Python packages
- `BMDS2003_HeartDisease_Data_PreparationAndModelling.ipynb`: Data preparation and modeling notebook
- `BMDS2003_HeartDisease_DataUnderstanding.ipynb`: Data exploration notebook
- `classification.png`, `confusion.png`, `rocauc.png`, `heart.png`, `columbia.jpg`: Images

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   python app.py
   ```
3. Open your browser and follow the instructions to input data and view predictions.

## Requirements

- Python 3.x
- See `requirements.txt` for package details

## License

This project is for educational purposes and to fulfill the assignment requirements of BMDS2003 Data Science. Created by Jin Yuan, Patricia Lee, Tian Li, and Cheong Kai Xin

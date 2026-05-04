# Sprint 2 Evidence Summary - C2 Smart Agriculture Platform

## Completed Work

### 1. Sensor Data Validation
Implemented sensor data validation for the `sensor_readings` table. The validation checks required columns, missing values, allowed sensor parameters, and safe value ranges.

Evidence:
- Great Expectations validation report
- Airflow validation task logs

### 2. Isolation Forest Anomaly Detection
Implemented Model 4 Sensor Anomaly Detection using Isolation Forest. One model was trained per sensor parameter.

Sensor parameters:
- soil_moisture
- soil_temp
- ambient_temp
- humidity
- pressure
- solar_radiation

Evidence:
- Saved `.joblib` model files
- Airflow train task logs
- MLflow training run

### 3. Anomaly Scoring and Database Storage
Implemented anomaly scoring for recent sensor readings. Detected anomalies are inserted into the PostgreSQL `anomaly_events` table.

Evidence:
- PostgreSQL anomaly_events query screenshot
- Airflow score task logs

### 4. Evaluation with Labelled Data
Created labelled anomaly data and evaluated the model using precision, recall, and F1-score.

Evidence:
- anomaly_detection_test_report.csv
- anomaly_predictions.csv
- MLflow evaluation run

### 5. MLflow Experiment Tracking
Logged training and evaluation runs under the `Sensor_Anomaly_Detection` experiment.

Evidence:
- MLflow experiment screenshot
- MLflow training run screenshot
- MLflow evaluation run screenshot
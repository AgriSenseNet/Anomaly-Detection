-- =========================================================
-- Sprint 2 Tables
-- =========================================================

-- Raw / processed sensor readings from C1
CREATE TABLE IF NOT EXISTS sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    field_id TEXT REFERENCES fields(field_id) ON DELETE CASCADE,
    device_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    parameter VARCHAR(50) NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    unit VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Detected anomaly events from Model 4
CREATE TABLE IF NOT EXISTS anomaly_events (
    id BIGSERIAL PRIMARY KEY,
    field_id TEXT REFERENCES fields(field_id) ON DELETE SET NULL,
    device_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    value DECIMAL(10,4),
    anomaly_score DECIMAL(10,6),
    anomaly_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Irrigation log table used later by irrigation optimization model
CREATE TABLE IF NOT EXISTS irrigation_log (
    id BIGSERIAL PRIMARY KEY,
    field_id TEXT REFERENCES fields(field_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    water_volume_mm DECIMAL(6,2),
    triggered_by VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Daily features table for ML model training
CREATE TABLE IF NOT EXISTS daily_features (
    id BIGSERIAL PRIMARY KEY,
    field_id TEXT REFERENCES fields(field_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    soil_moisture_mean DECIMAL(6,2),
    soil_moisture_std DECIMAL(6,2),
    soil_temp_mean DECIMAL(6,2),
    ambient_temp_mean DECIMAL(6,2),
    ambient_temp_min DECIMAL(6,2),
    ambient_temp_max DECIMAL(6,2),
    humidity_mean DECIMAL(6,2),
    pressure_mean DECIMAL(6,2),
    solar_lux_sum DECIMAL(12,2),
    gdd_accumulated DECIMAL(8,2),
    et0_nasa DECIMAL(6,2),
    rainfall_api DECIMAL(6,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(field_id, date)
);

-- =========================================================
-- Sprint 2 Indexes
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp
    ON sensor_readings(timestamp);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_field_id
    ON sensor_readings(field_id);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_parameter_time
    ON sensor_readings(parameter, timestamp);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_device_time
    ON sensor_readings(device_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_timestamp
    ON anomaly_events(timestamp);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_device_time
    ON anomaly_events(device_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_sensor_type
    ON anomaly_events(sensor_type);

CREATE INDEX IF NOT EXISTS idx_daily_features_field_date
    ON daily_features(field_id, date);
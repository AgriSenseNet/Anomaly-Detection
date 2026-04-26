-- Create fields table
CREATE TABLE IF NOT EXISTS fields (
    field_id TEXT PRIMARY KEY,
    crop_type TEXT,
    planting_date DATE,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    soil_ph DOUBLE PRECISION,
    field_capacity_vwc DOUBLE PRECISION
);

-- Create external_api_cache table
CREATE TABLE IF NOT EXISTS external_api_cache (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    field_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT fk_external_api_cache_field
        FOREIGN KEY (field_id)
        REFERENCES fields(field_id)
        ON DELETE CASCADE
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_external_api_cache_source
    ON external_api_cache(source);

CREATE INDEX IF NOT EXISTS idx_external_api_cache_field_id
    ON external_api_cache(field_id);

CREATE INDEX IF NOT EXISTS idx_external_api_cache_timestamp
    ON external_api_cache(timestamp);

CREATE INDEX IF NOT EXISTS idx_external_api_cache_expires_at
    ON external_api_cache(expires_at);

CREATE INDEX IF NOT EXISTS idx_fields_lat_lon
    ON fields(lat, lon);


    INSERT INTO fields (
    field_id,
    crop_type,
    planting_date,
    lat,
    lon,
    soil_ph,
    field_capacity_vwc
)
VALUES (
    'field_001',
    'tomato',
    '2026-04-20',
    7.2083,
    79.8358,
    NULL,
    35.0
);

INSERT INTO external_api_cache (
    source,
    field_id,
    data,
    expires_at
)
VALUES (
    'open_meteo',
    'field_001',
    '{
      "hourly": {
        "temperature_2m": [28.5],
        "relativehumidity_2m": [78],
        "precipitation": [0.0]
      }
    }'::jsonb,
    NOW() + INTERVAL '1 hour'
);
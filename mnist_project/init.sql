CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  predicted_digit INTEGER NOT NULL,
  confidence FLOAT NOT NULL,
  true_label INTEGER,
  image_data BYTEA,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

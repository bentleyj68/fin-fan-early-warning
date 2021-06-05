CREATE TABLE hvac_failures (
  id SERIAL PRIMARY KEY NOT NULL,
  primary_element TEXT,
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  duration TEXT,
  difference DECIMAL,
  comments TEXT,
  failure BOOLEAN
);

CREATE TABLE speed (
  id SERIAL PRIMARY KEY NOT NULL,
  equipment_id TEXT,
  speed INT
);
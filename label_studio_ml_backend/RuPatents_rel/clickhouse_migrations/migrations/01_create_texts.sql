CREATE OR REPLACE TABLE texts
(
    id UUID DEFAULT generateUUIDv4(),
    content String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY id


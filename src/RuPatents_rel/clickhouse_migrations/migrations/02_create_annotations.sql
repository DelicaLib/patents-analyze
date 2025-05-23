CREATE OR REPLACE TABLE annotations
(
    id UUID DEFAULT generateUUIDv4(),
    text_id Nullable(UUID),
    url Nullable(String),
    created_at DateTime DEFAULT now(),
    CONSTRAINT at_least_one_not_null CHECK (url IS NOT NULL OR text_id IS NOT NULL)
) ENGINE = MergeTree()
ORDER BY id
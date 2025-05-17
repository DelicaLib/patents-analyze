CREATE OR REPLACE TABLE annotations
(
    id UUID DEFAULT generateUUIDv4(),
    text_id UUID,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY id
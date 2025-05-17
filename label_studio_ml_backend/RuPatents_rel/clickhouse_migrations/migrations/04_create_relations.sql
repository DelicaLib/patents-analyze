CREATE OR REPLACE TABLE relations
(
    id UUID DEFAULT generateUUIDv4(),
    from_id UInt64,
    to_id UInt64,
    annotation_id UUID DEFAULT generateUUIDv4(),
    labels Array(String),
    direction String
) ENGINE = MergeTree()
ORDER BY id
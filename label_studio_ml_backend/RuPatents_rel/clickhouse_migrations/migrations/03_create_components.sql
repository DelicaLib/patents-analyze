CREATE OR REPLACE TABLE components
(
    id UUID DEFAULT generateUUIDv4(),
    annotation_id UUID,
    token_id String,
    start UInt32,
    end UInt32,
    labels Array(String),
    text String
) ENGINE = MergeTree()
ORDER BY (annotation_id, id)
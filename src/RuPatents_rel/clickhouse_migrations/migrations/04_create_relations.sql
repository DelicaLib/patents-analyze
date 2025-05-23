CREATE OR REPLACE TABLE relations
(
    id UUID DEFAULT generateUUIDv4(),
    from_id String,
    to_id String,
    annotation_id UUID,
    labels Array(String),
    direction String
) ENGINE = MergeTree()
ORDER BY (annotation_id, id)
CREATE OR REPLACE TABLE components
(
    id UInt64,
    annotation_id UUID,
    start UInt32,
    end UInt32,
    labels Array(String),
    text String
) ENGINE = MergeTree()
ORDER BY id
import clickhouse_connect
import os

CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "dev")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "dev")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "dev")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))

client = clickhouse_connect.get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD,
    database=CLICKHOUSE_DB
)


def ensure_migrations_table():
    client.command("""
        CREATE TABLE IF NOT EXISTS __migrations (
            filename String,
            applied_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY applied_at
    """)


def get_applied_migrations():
    rows = client.query("SELECT filename FROM __migrations").result_rows
    return {row[0] for row in rows}


def apply_migrations():
    ensure_migrations_table()
    applied = get_applied_migrations()

    for fname in sorted(os.listdir('migrations')):
        if fname.endswith('.sql') and fname not in applied:
            with open(f'migrations/{fname}', 'r') as f:
                sql = f.read()
            client.command(sql)
            client.insert('__migrations', [ (fname,) ], column_names=['filename'])
            print(f"Applied {fname}")


if __name__ == "__main__":
    apply_migrations()

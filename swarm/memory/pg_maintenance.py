import os
import sys
import typing
import psycopg


def _normalize_dsn(url: typing.Optional[str]) -> typing.Optional[str]:
    if not url:
        return None
    return (
        url.replace("postgresql+psycopg://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
    )


def main() -> int:
    dsn = _normalize_dsn(os.getenv("VECTOR_DATABASE_URL") or os.getenv("VECTOR_DB_URL"))
    if not dsn:
        print("VECTOR_DATABASE_URL not set", file=sys.stderr)
        return 2
    try:
        # Use a short timeout to avoid long hangs in CI
        with psycopg.connect(dsn, autocommit=True, timeout=10) as conn:
            with conn.cursor() as cur:
                print("VACUUM (ANALYZE)...", flush=True)
                cur.execute("VACUUM (ANALYZE)")
                try:
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
                    if cur.fetchone():
                        cur.execute("ANALYZE")
                except Exception:
                    # best-effort, not fatal
                    pass
        print("Maintenance completed", flush=True)
        return 0
    except Exception as e:
        print(f"maintenance failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

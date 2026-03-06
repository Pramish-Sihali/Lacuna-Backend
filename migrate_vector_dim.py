"""
One-time migration: change vector columns from 768-dim to 1024-dim.

Run from Lacuna-backend/ with the venv active:
    python migrate_vector_dim.py

What it does:
1. NULLs out all existing embeddings (they were 768-dim and are incompatible)
2. DROPs the vector columns
3. Re-ADDs them as vector(1024)

Safe to re-run — checks the current dimension before making any changes.
"""
import asyncio
import sys

from sqlalchemy import text
from app.database import AsyncSessionLocal

TARGET_DIM = 1024

TABLES = [
    ("chunks",   "embedding"),
    ("concepts", "embedding"),
    ("claims",   "embedding"),
]


async def get_current_dim(session, table: str, column: str):
    """Return the current vector dimension for the given column, or None."""
    result = await session.execute(text("""
        SELECT atttypmod
        FROM   pg_attribute
        JOIN   pg_class ON pg_class.oid = pg_attribute.attrelid
        WHERE  pg_class.relname = :tbl
          AND  pg_attribute.attname = :col
          AND  pg_attribute.attnum  > 0
    """), {"tbl": table, "col": column})
    row = result.fetchone()
    if row is None:
        return None
    return row[0] if row[0] > 0 else None


async def migrate():
    async with AsyncSessionLocal() as session:
        print("Checking current vector dimensions…\n")

        needs_migration = []
        for table, col in TABLES:
            dim = await get_current_dim(session, table, col)
            status = "OK" if dim == TARGET_DIM else f"NEEDS MIGRATION (current: {dim})"
            print(f"  {table}.{col}: {status}")
            if dim != TARGET_DIM:
                needs_migration.append((table, col))

        if not needs_migration:
            print(f"\nAll columns already at vector({TARGET_DIM}). Nothing to do.")
            return

        print(f"\nMigrating {len(needs_migration)} column(s) to vector({TARGET_DIM})…\n")

        for table, col in needs_migration:
            print(f"  [{table}.{col}] Nulling existing embeddings…")
            await session.execute(text(f"UPDATE {table} SET {col} = NULL"))

            print(f"  [{table}.{col}] Dropping column…")
            await session.execute(text(f"ALTER TABLE {table} DROP COLUMN {col}"))

            print(f"  [{table}.{col}] Re-adding as vector({TARGET_DIM})…")
            await session.execute(text(
                f"ALTER TABLE {table} ADD COLUMN {col} vector({TARGET_DIM})"
            ))
            print(f"  [{table}.{col}] Done.\n")

        await session.commit()
        print("Migration complete.")
        print("Run the pipeline (process-all) to regenerate embeddings with the new model.")


if __name__ == "__main__":
    print("Lacuna — vector dimension migration\n" + "=" * 40)
    try:
        asyncio.run(migrate())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

_pool = None  # Global pool

async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 5432)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            ssl='require'  # Or use ssl=False for local dev
        )
    return _pool

async def query(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as connection:
        try:
            result = await connection.fetch(sql, *args)
            return [dict(record) for record in result]  # Convert to list of dicts
        except Exception as e:
            print(f"🛑 Database query error: {e}")
            raise

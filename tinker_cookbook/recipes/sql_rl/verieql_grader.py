"""
Async wrapper for VeriEQL-based SQL equivalence grading.

VeriEQL is run as a subprocess so its Z3 solver state is isolated from the
main async training process and can be killed cleanly on timeout.
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_HELPER_SCRIPT = os.path.join(os.path.dirname(__file__), "verieql_helper.py")

# SQLite affinity → VeriEQL type.
# VeriEQL supports INT, VARCHAR; we map everything else to one of these.
_TYPE_KEYWORDS: list[tuple[list[str], str]] = [
    (["INT", "INTEGER", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT"], "INT"),
    (["BOOL"], "INT"),
    (["REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "NUMBER"], "INT"),
    # DATE/DATETIME → VARCHAR: VeriEQL's DATE type has limited support
    (["DATE", "TIME", "YEAR"], "VARCHAR"),
    (["CHAR", "TEXT", "CLOB", "STRING", "VARCHAR", "NVARCHAR", "BLOB", "BINARY"], "VARCHAR"),
]


def _sqlite_type_to_verieql(sqlite_type: str) -> str:
    t = (sqlite_type or "").upper().strip()
    for keywords, mapped in _TYPE_KEYWORDS:
        if any(k in t for k in keywords):
            return mapped
    return "VARCHAR"  # safe default


@lru_cache(maxsize=None)
def extract_schema_from_sqlite(db_file: str) -> dict[str, dict[str, str]]:
    """Return {table_name: {column_name: verieql_type}} for all tables in the SQLite file."""
    schema: dict[str, dict[str, str]] = {}
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = {row[1]: _sqlite_type_to_verieql(row[2]) for row in cursor.fetchall()}
            if columns:
                schema[table] = columns
    finally:
        conn.close()
    return schema


async def grade_with_verieql(
    db_file: str,
    pred_sql: str,
    gold_sql: str,
    timeout: float = 30.0,
    bound_size: int = 2,
) -> Optional[bool]:
    """
    Check whether pred_sql and gold_sql are semantically equivalent using VeriEQL.

    VeriEQL is invoked in a subprocess so Z3 state cannot leak into the main process
    and the subprocess can be forcibly killed on timeout.

    Returns:
        True   -- formally equivalent (safe to grant reward 1.0)
        False  -- provably non-equivalent
        None   -- unknown (timeout, unsupported SQL feature, parse error, etc.)
    """
    try:
        schema = extract_schema_from_sqlite(db_file)
    except Exception as e:
        logger.debug("VeriEQL: schema extraction failed: %s", e)
        return None

    if not schema:
        return None

    payload = json.dumps({
        "schema": schema,
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "bound_size": bound_size,
    }).encode()

    proc = await asyncio.create_subprocess_exec(
        sys.executable, _HELPER_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(payload),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        logger.info("VeriEQL: timed out after %.0fs for pred=%s", timeout, pred_sql[:80])
        return None
    except Exception as e:
        logger.info("VeriEQL: subprocess error: %s", e)
        return None

    try:
        out = json.loads(stdout.decode().strip())
    except Exception as e:
        logger.info("VeriEQL: could not parse output: %s (stdout=%r)", e, stdout[:200])
        return None

    result = out.get("result")
    if result is None and "error" in out:
        logger.info("VeriEQL: %s", out["error"])
    return result if isinstance(result, bool) else None

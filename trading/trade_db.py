"""
SQLite (dev) / PostgreSQL (production) trade database.

Auto-detects DATABASE_URL environment variable:
  - If set  → PostgreSQL — data persists across Railway deploys
  - If not  → SQLite at trading/data/trading.db (local dev only)

To enable persistence on Railway:
  1. Add the Railway PostgreSQL plugin to your project (or use any Postgres URL)
  2. Railway automatically sets DATABASE_URL in your environment
  3. Redeploy — the bot creates the schema and retains all data between deploys

psycopg2-binary is listed in requirements.txt but only imported when DATABASE_URL
is present, so SQLite-only setups don't need the extra package installed.
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
DB_PATH: str = os.getenv(
    "TRADE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "trading.db"),
)

# Try to load psycopg2 only when a Postgres URL is configured.
_POSTGRES_OK = False
if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
        _POSTGRES_OK = True
    except ImportError:
        import logging as _log
        _log.getLogger(__name__).warning(
            "DATABASE_URL is set but psycopg2 is not installed — "
            "falling back to SQLite (data will reset on redeploy)"
        )
        DATABASE_URL = None


def _pg() -> bool:
    """True when PostgreSQL is the active backend."""
    return bool(DATABASE_URL and _POSTGRES_OK)


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _ph() -> str:
    """Placeholder character for the active backend."""
    return "%s" if _pg() else "?"


def _adapt_sql(sql: str) -> str:
    """Convert SQLite-style ? placeholders to %s for PostgreSQL."""
    return sql.replace("?", "%s") if _pg() else sql


# ── Schema ────────────────────────────────────────────────────────────────────

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    units           REAL NOT NULL DEFAULT 0,
    entry_price     REAL NOT NULL DEFAULT 0,
    exit_price      REAL,
    stop_loss       REAL,
    take_profit     REAL,
    confidence      REAL NOT NULL DEFAULT 0,
    reasoning       TEXT DEFAULT '',
    outcome         TEXT NOT NULL DEFAULT 'open',
    pnl             REAL NOT NULL DEFAULT 0,
    opened_at       TEXT NOT NULL,
    closed_at       TEXT,
    lesson_category TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id   TEXT PRIMARY KEY,
    trade_id    TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT '',
    description TEXT DEFAULT '',
    rule_added  TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_stats (
    date          TEXT PRIMARY KEY,
    total_trades  INTEGER NOT NULL DEFAULT 0,
    wins          INTEGER NOT NULL DEFAULT 0,
    losses        INTEGER NOT NULL DEFAULT 0,
    pnl           REAL NOT NULL DEFAULT 0,
    max_drawdown  REAL NOT NULL DEFAULT 0,
    balance       REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS signals (
    rowid_pk       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0,
    entry          REAL NOT NULL DEFAULT 0,
    sl             REAL,
    tp             REAL,
    taken          INTEGER NOT NULL DEFAULT 0,
    reason_skipped TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS positions (
    trade_id    TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    units       REAL NOT NULL DEFAULT 0,
    entry_price REAL NOT NULL DEFAULT 0,
    stop_loss   REAL,
    take_profit REAL,
    opened_at   TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

_PG_TABLES = [
    """CREATE TABLE IF NOT EXISTS trades (
        trade_id        TEXT PRIMARY KEY,
        symbol          TEXT NOT NULL,
        side            TEXT NOT NULL,
        units           REAL NOT NULL DEFAULT 0,
        entry_price     REAL NOT NULL DEFAULT 0,
        exit_price      REAL,
        stop_loss       REAL,
        take_profit     REAL,
        confidence      REAL NOT NULL DEFAULT 0,
        reasoning       TEXT DEFAULT '',
        outcome         TEXT NOT NULL DEFAULT 'open',
        pnl             REAL NOT NULL DEFAULT 0,
        opened_at       TEXT NOT NULL,
        closed_at       TEXT,
        lesson_category TEXT DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS lessons (
        lesson_id   TEXT PRIMARY KEY,
        trade_id    TEXT NOT NULL,
        category    TEXT NOT NULL DEFAULT '',
        description TEXT DEFAULT '',
        rule_added  TEXT DEFAULT '',
        created_at  TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS daily_stats (
        date          TEXT PRIMARY KEY,
        total_trades  INTEGER NOT NULL DEFAULT 0,
        wins          INTEGER NOT NULL DEFAULT 0,
        losses        INTEGER NOT NULL DEFAULT 0,
        pnl           REAL NOT NULL DEFAULT 0,
        max_drawdown  REAL NOT NULL DEFAULT 0,
        balance       REAL NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS signals (
        id             SERIAL PRIMARY KEY,
        timestamp      TEXT NOT NULL,
        symbol         TEXT NOT NULL,
        side           TEXT NOT NULL,
        confidence     REAL NOT NULL DEFAULT 0,
        entry          REAL NOT NULL DEFAULT 0,
        sl             REAL,
        tp             REAL,
        taken          INTEGER NOT NULL DEFAULT 0,
        reason_skipped TEXT DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS positions (
        trade_id    TEXT PRIMARY KEY,
        symbol      TEXT NOT NULL,
        side        TEXT NOT NULL,
        units       REAL NOT NULL DEFAULT 0,
        entry_price REAL NOT NULL DEFAULT 0,
        stop_loss   REAL,
        take_profit REAL,
        opened_at   TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )""",
]


class TradeDB:
    """
    Lightweight database wrapper for all trading persistence.

    The public API is identical regardless of whether PostgreSQL or SQLite
    is in use — callers never need to know which backend is active.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        if _pg():
            self._init_pg()
        else:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._init_sqlite()
            self._run_sqlite_migrations()

    # ── Connections ───────────────────────────────────────────────────────────

    def _connect_pg(self):
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        return conn

    def _connect_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Core executor ─────────────────────────────────────────────────────────

    def _execute(
        self,
        sql: str,
        params: tuple = (),
        *,
        fetch: Optional[str] = None,
        commit: bool = True,
    ) -> Any:
        """Execute SQL on whichever backend is active."""
        if _pg():
            return self._exec_pg(sql, params, fetch=fetch, commit=commit)
        return self._exec_sqlite(sql, params, fetch=fetch, commit=commit)

    def _exec_pg(self, sql: str, params: tuple, *, fetch, commit) -> Any:
        conn = self._connect_pg()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                if commit:
                    conn.commit()
                if fetch == "one":
                    row = cur.fetchone()
                    return dict(row) if row else None
                if fetch == "all":
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _exec_sqlite(self, sql: str, params: tuple, *, fetch, commit) -> Any:
        conn = self._connect_sqlite()
        try:
            cur = conn.execute(sql, params)
            if commit:
                conn.commit()
            if fetch == "one":
                row = cur.fetchone()
                return dict(row) if row else None
            if fetch == "all":
                return [dict(r) for r in cur.fetchall()]
            return cur.lastrowid
        finally:
            conn.close()

    # ── Schema init ───────────────────────────────────────────────────────────

    def _init_pg(self) -> None:
        conn = self._connect_pg()
        try:
            with conn.cursor() as cur:
                for stmt in _PG_TABLES:
                    cur.execute(stmt)
            conn.commit()
        finally:
            conn.close()

    def _init_sqlite(self) -> None:
        conn = self._connect_sqlite()
        try:
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _run_sqlite_migrations(self) -> None:
        """Idempotent schema migrations for SQLite only."""
        conn = self._connect_sqlite()
        try:
            self._sqlite_drop_lessons_fk(conn)
            conn.commit()
        finally:
            conn.close()

    def _column_exists(self, conn: sqlite3.Connection, table: str, col: str) -> bool:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(row["name"] == col for row in cur.fetchall())

    def _sqlite_drop_lessons_fk(self, conn: sqlite3.Connection) -> None:
        """Remove FOREIGN KEY from lessons table if still present."""
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='lessons'"
            ).fetchone()
            if row is None or "FOREIGN KEY" not in (row[0] or ""):
                return
            conn.execute("ALTER TABLE lessons RENAME TO lessons_old")
            conn.execute("""
                CREATE TABLE lessons (
                    lesson_id TEXT PRIMARY KEY, trade_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '', description TEXT DEFAULT '',
                    rule_added TEXT DEFAULT '', created_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO lessons SELECT lesson_id, trade_id, category, "
                "description, rule_added, created_at FROM lessons_old"
            )
            conn.execute("DROP TABLE lessons_old")
            conn.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Lessons FK migration skipped: {e}")

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    @staticmethod
    def _to_list(rows) -> list[dict]:
        if not rows:
            return []
        return [dict(r) for r in rows] if not isinstance(rows[0], dict) else list(rows)

    # ── Trades ────────────────────────────────────────────────────────────────

    def save_trade(self, trade: dict = None, **kwargs) -> None:
        if trade is None:
            trade = kwargs
        t = (
            trade["trade_id"], trade["symbol"], trade["side"],
            trade.get("units", 0), trade.get("entry_price", 0),
            trade.get("exit_price"), trade.get("stop_loss"), trade.get("take_profit"),
            trade.get("confidence", 0), trade.get("reasoning", ""),
            trade.get("outcome", "open"), trade.get("pnl", 0),
            trade.get("opened_at", self._now()), trade.get("closed_at"),
            trade.get("lesson_category", ""),
        )
        cols = ("trade_id,symbol,side,units,entry_price,exit_price,stop_loss,"
                "take_profit,confidence,reasoning,outcome,pnl,opened_at,closed_at,lesson_category")
        ph_list = ",".join([_ph()] * 15)
        if _pg():
            self._execute(
                f"INSERT INTO trades ({cols}) VALUES ({ph_list}) "
                "ON CONFLICT (trade_id) DO UPDATE SET "
                "symbol=EXCLUDED.symbol,side=EXCLUDED.side,units=EXCLUDED.units,"
                "entry_price=EXCLUDED.entry_price,exit_price=EXCLUDED.exit_price,"
                "stop_loss=EXCLUDED.stop_loss,take_profit=EXCLUDED.take_profit,"
                "confidence=EXCLUDED.confidence,reasoning=EXCLUDED.reasoning,"
                "outcome=EXCLUDED.outcome,pnl=EXCLUDED.pnl,"
                "opened_at=EXCLUDED.opened_at,closed_at=EXCLUDED.closed_at,"
                "lesson_category=EXCLUDED.lesson_category",
                t,
            )
        else:
            keys = cols.split(",")
            named = dict(zip(keys, t))
            self._execute(
                f"INSERT OR REPLACE INTO trades ({cols}) VALUES "
                f"(:{','.join(f':{k}' for k in keys)[1:]})",
                named,
            )

    def update_trade(self, trade_id: str = None, updates: dict = None, **kwargs) -> None:
        if updates is None:
            updates = kwargs
        updates.pop("trade_id", None)
        if not updates:
            return
        ph = _ph()
        if _pg():
            set_clause = ", ".join(f"{k} = {ph}" for k in updates)
            params = tuple(updates.values()) + (trade_id,)
            self._execute(f"UPDATE trades SET {set_clause} WHERE trade_id = {ph}", params)
        else:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            updates["trade_id"] = trade_id
            self._execute(f"UPDATE trades SET {set_clause} WHERE trade_id = :trade_id", updates)

    def get_trade(self, trade_id: str) -> Optional[dict]:
        ph = _ph()
        return self._execute(
            f"SELECT * FROM trades WHERE trade_id = {ph}", (trade_id,),
            fetch="one", commit=False,
        )

    def get_open_trades(self) -> list[dict]:
        rows = self._execute(
            "SELECT * FROM trades WHERE outcome = 'open' ORDER BY opened_at DESC",
            fetch="all", commit=False,
        )
        return rows or []

    def get_all_trades(self, symbol=None, outcome=None, limit=500) -> list[dict]:
        clauses, params = [], []
        ph = _ph()
        if symbol:
            clauses.append(f"symbol = {ph}"); params.append(symbol)
        if outcome:
            clauses.append(f"outcome = {ph}"); params.append(outcome)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._execute(
            f"SELECT * FROM trades{where} ORDER BY opened_at DESC LIMIT {ph}",
            tuple(params), fetch="all", commit=False,
        )
        return rows or []

    # ── Lessons ───────────────────────────────────────────────────────────────

    def save_lesson(self, lesson: dict = None, **kwargs) -> None:
        if lesson is None:
            lesson = kwargs
        vals = (
            lesson.get("lesson_id", ""), lesson.get("trade_id", ""),
            lesson.get("category", ""), lesson.get("description", ""),
            lesson.get("rule_added", ""), lesson.get("created_at", self._now()),
        )
        cols = "lesson_id,trade_id,category,description,rule_added,created_at"
        ph_list = ",".join([_ph()] * 6)
        if _pg():
            self._execute(
                f"INSERT INTO lessons ({cols}) VALUES ({ph_list}) "
                "ON CONFLICT (lesson_id) DO UPDATE SET "
                "category=EXCLUDED.category,description=EXCLUDED.description,"
                "rule_added=EXCLUDED.rule_added",
                vals,
            )
        else:
            self._execute(
                f"INSERT OR REPLACE INTO lessons ({cols}) VALUES ({ph_list})", vals
            )

    def get_lessons(self, category=None, trade_id=None, limit=200) -> list[dict]:
        clauses, params = [], []
        ph = _ph()
        if category:
            clauses.append(f"category = {ph}"); params.append(category)
        if trade_id:
            clauses.append(f"trade_id = {ph}"); params.append(trade_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._execute(
            f"SELECT * FROM lessons{where} ORDER BY created_at DESC LIMIT {ph}",
            tuple(params), fetch="all", commit=False,
        )
        return rows or []

    # ── Daily stats ───────────────────────────────────────────────────────────

    def save_daily_stats(self, stats: dict = None, **kwargs) -> None:
        if stats is None:
            stats = kwargs
        vals = (
            stats.get("date", self._now()[:10]),
            stats.get("total_trades", 0), stats.get("wins", 0),
            stats.get("losses", 0), stats.get("pnl", 0),
            stats.get("max_drawdown", 0), stats.get("balance", 0),
        )
        cols = "date,total_trades,wins,losses,pnl,max_drawdown,balance"
        ph_list = ",".join([_ph()] * 7)
        if _pg():
            self._execute(
                f"INSERT INTO daily_stats ({cols}) VALUES ({ph_list}) "
                "ON CONFLICT (date) DO UPDATE SET "
                "total_trades=EXCLUDED.total_trades,wins=EXCLUDED.wins,"
                "losses=EXCLUDED.losses,pnl=EXCLUDED.pnl,"
                "max_drawdown=EXCLUDED.max_drawdown,balance=EXCLUDED.balance",
                vals,
            )
        else:
            self._execute(
                f"INSERT OR REPLACE INTO daily_stats ({cols}) VALUES ({ph_list})", vals
            )

    def get_daily_stats(self, start_date=None, end_date=None) -> list[dict]:
        clauses, params = [], []
        ph = _ph()
        if start_date:
            clauses.append(f"date >= {ph}"); params.append(start_date)
        if end_date:
            clauses.append(f"date <= {ph}"); params.append(end_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._execute(
            f"SELECT * FROM daily_stats{where} ORDER BY date DESC",
            tuple(params), fetch="all", commit=False,
        )
        return rows or []

    # ── Signals ───────────────────────────────────────────────────────────────

    def save_signal(self, signal: dict = None, **kwargs) -> None:
        if signal is None:
            signal = kwargs
        vals = (
            signal.get("timestamp", self._now()), signal.get("symbol", ""),
            signal.get("side", ""), signal.get("confidence", 0),
            signal.get("entry", 0), signal.get("sl"), signal.get("tp"),
            1 if signal.get("taken") else 0, signal.get("reason_skipped", ""),
        )
        cols = "timestamp,symbol,side,confidence,entry,sl,tp,taken,reason_skipped"
        ph_list = ",".join([_ph()] * 9)
        self._execute(f"INSERT INTO signals ({cols}) VALUES ({ph_list})", vals)

    def get_signals(self, symbol=None, taken=None, limit=200) -> list[dict]:
        clauses, params = [], []
        ph = _ph()
        if symbol:
            clauses.append(f"symbol = {ph}"); params.append(symbol)
        if taken is not None:
            clauses.append(f"taken = {ph}"); params.append(1 if taken else 0)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._execute(
            f"SELECT * FROM signals{where} ORDER BY timestamp DESC LIMIT {ph}",
            tuple(params), fetch="all", commit=False,
        )
        results = rows or []
        for r in results:
            r["taken"] = bool(r.get("taken"))
        return results

    # ── Positions ─────────────────────────────────────────────────────────────

    def save_position(self, pos: dict) -> None:
        vals = (
            pos["trade_id"], pos["symbol"], pos["side"],
            pos.get("units", 0), pos.get("entry_price", 0),
            pos.get("stop_loss"), pos.get("take_profit"),
            pos.get("opened_at", self._now()), self._now(),
        )
        cols = "trade_id,symbol,side,units,entry_price,stop_loss,take_profit,opened_at,updated_at"
        ph_list = ",".join([_ph()] * 9)
        if _pg():
            self._execute(
                f"INSERT INTO positions ({cols}) VALUES ({ph_list}) "
                "ON CONFLICT (trade_id) DO UPDATE SET "
                "units=EXCLUDED.units,stop_loss=EXCLUDED.stop_loss,"
                "take_profit=EXCLUDED.take_profit,updated_at=EXCLUDED.updated_at",
                vals,
            )
        else:
            self._execute(
                f"INSERT OR REPLACE INTO positions ({cols}) VALUES ({ph_list})", vals
            )

    def remove_position(self, trade_id: str) -> None:
        ph = _ph()
        self._execute(f"DELETE FROM positions WHERE trade_id = {ph}", (trade_id,))

    def get_positions(self) -> list[dict]:
        rows = self._execute(
            "SELECT * FROM positions ORDER BY opened_at DESC",
            fetch="all", commit=False,
        )
        return rows or []

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_win_rate(self, symbol=None) -> dict:
        where = "WHERE outcome IN ('win', 'loss')"
        params: tuple = ()
        ph = _ph()
        if symbol:
            where += f" AND symbol = {ph}"
            params = (symbol,)
        row = self._execute(
            f"SELECT COUNT(*) AS total, "
            f"SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS wins, "
            f"SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) AS losses "
            f"FROM trades {where}",
            params, fetch="one", commit=False,
        ) or {}
        total = row.get("total") or 0
        row["win_rate"] = round((row.get("wins") or 0) / total, 4) if total else 0.0
        return row

    def get_pnl_by_symbol(self) -> list[dict]:
        rows = self._execute(
            "SELECT symbol, COUNT(*) AS trades, "
            "SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS wins, "
            "SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) AS losses, "
            "ROUND(CAST(SUM(pnl) AS NUMERIC),4) AS total_pnl, "
            "ROUND(CAST(AVG(pnl) AS NUMERIC),4) AS avg_pnl "
            "FROM trades WHERE outcome IN ('win','loss') "
            "GROUP BY symbol ORDER BY total_pnl DESC",
            fetch="all", commit=False,
        )
        return rows or []

    def get_performance_summary(self) -> dict:
        row = self._execute(
            "SELECT COUNT(*) AS total_trades, "
            "SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS wins, "
            "SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) AS losses, "
            "ROUND(CAST(SUM(pnl) AS NUMERIC),4) AS total_pnl, "
            "ROUND(CAST(AVG(pnl) AS NUMERIC),4) AS avg_pnl, "
            "ROUND(CAST(MAX(pnl) AS NUMERIC),4) AS best_trade, "
            "ROUND(CAST(MIN(pnl) AS NUMERIC),4) AS worst_trade, "
            "ROUND(CAST(AVG(confidence) AS NUMERIC),4) AS avg_confidence "
            "FROM trades WHERE outcome IN ('win','loss')",
            fetch="one", commit=False,
        ) or {}
        total = row.get("total_trades") or 0
        row["win_rate"] = round((row.get("wins") or 0) / total, 4) if total else 0.0

        pf = self._execute(
            "SELECT COALESCE(SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END),0) AS gross_win, "
            "COALESCE(SUM(CASE WHEN pnl<0 THEN ABS(pnl) ELSE 0 END),0) AS gross_loss "
            "FROM trades WHERE outcome IN ('win','loss')",
            fetch="one", commit=False,
        ) or {}
        gl = pf.get("gross_loss") or 0
        row["profit_factor"] = round(pf.get("gross_win", 0) / gl, 4) if gl else float("inf")

        open_row = self._execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE outcome='open'",
            fetch="one", commit=False,
        ) or {}
        row["open_trades"] = open_row.get("cnt", 0)
        return row

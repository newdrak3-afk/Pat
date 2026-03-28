"""
SQLite-based trade database — replaces JSON file storage.

Thread-safe: every public method opens its own connection and closes it
before returning, so callers on different threads never share a handle.
"""

import sqlite3
import os
import threading
from datetime import datetime, date
from typing import Any, Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "data", "trading.db")


class TradeDB:
    """Lightweight SQLite wrapper for all trading persistence."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()
        self._run_migrations()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with row-factory set to sqlite3.Row."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _execute(
        self,
        sql: str,
        params: tuple = (),
        *,
        fetch: Optional[str] = None,
        commit: bool = True,
    ) -> Any:
        """Execute *sql* with a fresh connection.

        Parameters
        ----------
        fetch : None | "one" | "all"
            What to return after executing.
        commit : bool
            Whether to commit (writes) — selects don't need it.
        """
        conn = self._connect()
        try:
            cur = conn.execute(sql, params)
            if commit:
                conn.commit()
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return cur.lastrowid
        finally:
            conn.close()

    def _executemany(self, sql: str, seq: list[tuple]) -> None:
        conn = self._connect()
        try:
            conn.executemany(sql, seq)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema — tables
    # ------------------------------------------------------------------

    _TABLES_SQL = """
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
        created_at  TEXT NOT NULL,
        FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
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

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(self._TABLES_SQL)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Migrations — add columns if they don't exist yet
    # ------------------------------------------------------------------

    def _column_exists(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(row["name"] == column for row in cur.fetchall())

    def _add_column_if_missing(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        col_type: str,
        default: str = "",
    ) -> None:
        if not self._column_exists(conn, table, column):
            default_clause = f" DEFAULT {default}" if default else ""
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
            )

    def _run_migrations(self) -> None:
        """Add any new columns that were introduced after the initial schema.

        Each migration is idempotent — safe to re-run.
        """
        conn = self._connect()
        try:
            # Example future migrations:
            # self._add_column_if_missing(conn, "trades", "strategy", "TEXT", "''")
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
        return [dict(r) for r in rows]

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(self, trade: dict) -> None:
        """Insert or replace a trade row.

        ``trade`` must contain at least *trade_id*, *symbol*, and *side*.
        Missing optional keys are filled with sensible defaults.
        """
        t = {
            "trade_id": trade["trade_id"],
            "symbol": trade["symbol"],
            "side": trade["side"],
            "units": trade.get("units", 0),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price"),
            "stop_loss": trade.get("stop_loss"),
            "take_profit": trade.get("take_profit"),
            "confidence": trade.get("confidence", 0),
            "reasoning": trade.get("reasoning", ""),
            "outcome": trade.get("outcome", "open"),
            "pnl": trade.get("pnl", 0),
            "opened_at": trade.get("opened_at", self._now()),
            "closed_at": trade.get("closed_at"),
            "lesson_category": trade.get("lesson_category", ""),
        }
        self._execute(
            """
            INSERT OR REPLACE INTO trades
                (trade_id, symbol, side, units, entry_price, exit_price,
                 stop_loss, take_profit, confidence, reasoning, outcome,
                 pnl, opened_at, closed_at, lesson_category)
            VALUES
                (:trade_id, :symbol, :side, :units, :entry_price, :exit_price,
                 :stop_loss, :take_profit, :confidence, :reasoning, :outcome,
                 :pnl, :opened_at, :closed_at, :lesson_category)
            """,
            t,
        )

    def update_trade(self, trade_id: str, updates: dict) -> None:
        """Update specific columns of an existing trade."""
        if not updates:
            return
        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        updates["trade_id"] = trade_id
        self._execute(
            f"UPDATE trades SET {set_clause} WHERE trade_id = :trade_id",
            updates,
        )

    def get_trade(self, trade_id: str) -> Optional[dict]:
        row = self._execute(
            "SELECT * FROM trades WHERE trade_id = ?",
            (trade_id,),
            fetch="one",
            commit=False,
        )
        return self._row_to_dict(row)

    def get_open_trades(self) -> list[dict]:
        rows = self._execute(
            "SELECT * FROM trades WHERE outcome = 'open' ORDER BY opened_at DESC",
            fetch="all",
            commit=False,
        )
        return self._rows_to_dicts(rows)

    def get_all_trades(
        self,
        symbol: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._execute(
            f"SELECT * FROM trades{where} ORDER BY opened_at DESC LIMIT ?",
            tuple(params),
            fetch="all",
            commit=False,
        )
        return self._rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def save_lesson(self, lesson: dict) -> None:
        l = {
            "lesson_id": lesson["lesson_id"],
            "trade_id": lesson["trade_id"],
            "category": lesson.get("category", ""),
            "description": lesson.get("description", ""),
            "rule_added": lesson.get("rule_added", ""),
            "created_at": lesson.get("created_at", self._now()),
        }
        self._execute(
            """
            INSERT OR REPLACE INTO lessons
                (lesson_id, trade_id, category, description, rule_added, created_at)
            VALUES
                (:lesson_id, :trade_id, :category, :description, :rule_added, :created_at)
            """,
            l,
        )

    def get_lessons(
        self,
        category: Optional[str] = None,
        trade_id: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if trade_id:
            clauses.append("trade_id = ?")
            params.append(trade_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._execute(
            f"SELECT * FROM lessons{where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
            fetch="all",
            commit=False,
        )
        return self._rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Daily stats
    # ------------------------------------------------------------------

    def save_daily_stats(self, stats: dict) -> None:
        s = {
            "date": stats["date"],
            "total_trades": stats.get("total_trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "pnl": stats.get("pnl", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "balance": stats.get("balance", 0),
        }
        self._execute(
            """
            INSERT OR REPLACE INTO daily_stats
                (date, total_trades, wins, losses, pnl, max_drawdown, balance)
            VALUES
                (:date, :total_trades, :wins, :losses, :pnl, :max_drawdown, :balance)
            """,
            s,
        )

    def get_daily_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._execute(
            f"SELECT * FROM daily_stats{where} ORDER BY date DESC",
            tuple(params),
            fetch="all",
            commit=False,
        )
        return self._rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def save_signal(self, signal: dict) -> None:
        s = {
            "timestamp": signal.get("timestamp", self._now()),
            "symbol": signal["symbol"],
            "side": signal["side"],
            "confidence": signal.get("confidence", 0),
            "entry": signal.get("entry", 0),
            "sl": signal.get("sl"),
            "tp": signal.get("tp"),
            "taken": 1 if signal.get("taken") else 0,
            "reason_skipped": signal.get("reason_skipped", ""),
        }
        self._execute(
            """
            INSERT INTO signals
                (timestamp, symbol, side, confidence, entry, sl, tp, taken, reason_skipped)
            VALUES
                (:timestamp, :symbol, :side, :confidence, :entry, :sl, :tp, :taken, :reason_skipped)
            """,
            s,
        )

    def get_signals(
        self,
        symbol: Optional[str] = None,
        taken: Optional[bool] = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if taken is not None:
            clauses.append("taken = ?")
            params.append(1 if taken else 0)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._execute(
            f"SELECT * FROM signals{where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
            fetch="all",
            commit=False,
        )
        results = self._rows_to_dicts(rows)
        # Convert the integer back to a bool for callers.
        for r in results:
            r["taken"] = bool(r["taken"])
        return results

    # ------------------------------------------------------------------
    # Positions (convenience mirror of open trades for fast lookup)
    # ------------------------------------------------------------------

    def save_position(self, pos: dict) -> None:
        p = {
            "trade_id": pos["trade_id"],
            "symbol": pos["symbol"],
            "side": pos["side"],
            "units": pos.get("units", 0),
            "entry_price": pos.get("entry_price", 0),
            "stop_loss": pos.get("stop_loss"),
            "take_profit": pos.get("take_profit"),
            "opened_at": pos.get("opened_at", self._now()),
            "updated_at": self._now(),
        }
        self._execute(
            """
            INSERT OR REPLACE INTO positions
                (trade_id, symbol, side, units, entry_price, stop_loss,
                 take_profit, opened_at, updated_at)
            VALUES
                (:trade_id, :symbol, :side, :units, :entry_price, :stop_loss,
                 :take_profit, :opened_at, :updated_at)
            """,
            p,
        )

    def remove_position(self, trade_id: str) -> None:
        self._execute("DELETE FROM positions WHERE trade_id = ?", (trade_id,))

    def get_positions(self) -> list[dict]:
        rows = self._execute(
            "SELECT * FROM positions ORDER BY opened_at DESC",
            fetch="all",
            commit=False,
        )
        return self._rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_win_rate(self, symbol: Optional[str] = None) -> dict:
        """Return win-rate stats: {total, wins, losses, win_rate}."""
        where = "WHERE outcome IN ('win', 'loss')"
        params: tuple = ()
        if symbol:
            where += " AND symbol = ?"
            params = (symbol,)
        row = self._execute(
            f"""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)  AS wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses
            FROM trades
            {where}
            """,
            params,
            fetch="one",
            commit=False,
        )
        d = self._row_to_dict(row) or {"total": 0, "wins": 0, "losses": 0}
        total = d["total"] or 0
        d["win_rate"] = round(d["wins"] / total, 4) if total else 0.0
        return d

    def get_pnl_by_symbol(self) -> list[dict]:
        """Return aggregate PnL grouped by symbol."""
        rows = self._execute(
            """
            SELECT
                symbol,
                COUNT(*)                                           AS trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)  AS wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
                ROUND(SUM(pnl), 4)                                 AS total_pnl,
                ROUND(AVG(pnl), 4)                                 AS avg_pnl
            FROM trades
            WHERE outcome IN ('win', 'loss')
            GROUP BY symbol
            ORDER BY total_pnl DESC
            """,
            fetch="all",
            commit=False,
        )
        return self._rows_to_dicts(rows)

    def get_performance_summary(self) -> dict:
        """High-level performance snapshot across all closed trades."""
        row = self._execute(
            """
            SELECT
                COUNT(*)                                           AS total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)  AS wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
                ROUND(SUM(pnl), 4)                                 AS total_pnl,
                ROUND(AVG(pnl), 4)                                 AS avg_pnl,
                ROUND(MAX(pnl), 4)                                 AS best_trade,
                ROUND(MIN(pnl), 4)                                 AS worst_trade,
                ROUND(AVG(confidence), 4)                          AS avg_confidence
            FROM trades
            WHERE outcome IN ('win', 'loss')
            """,
            fetch="one",
            commit=False,
        )
        d = self._row_to_dict(row) or {}
        total = d.get("total_trades") or 0
        d["win_rate"] = round(d["wins"] / total, 4) if total else 0.0

        # Profit factor: gross wins / gross losses
        pf_row = self._execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) AS gross_win,
                COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) AS gross_loss
            FROM trades
            WHERE outcome IN ('win', 'loss')
            """,
            fetch="one",
            commit=False,
        )
        pf = self._row_to_dict(pf_row) or {"gross_win": 0, "gross_loss": 0}
        gross_loss = pf["gross_loss"] or 0
        d["profit_factor"] = (
            round(pf["gross_win"] / gross_loss, 4) if gross_loss else float("inf")
        )

        # Open-position count
        open_row = self._execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE outcome = 'open'",
            fetch="one",
            commit=False,
        )
        d["open_trades"] = (self._row_to_dict(open_row) or {}).get("cnt", 0)

        return d

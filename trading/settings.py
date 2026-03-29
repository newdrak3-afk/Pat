"""
Master Settings — Toggle features on/off from one place.

Controls what runs and what doesn't:
- Scanning (forex pairs)
- Auto-trading (place real orders)
- Demo mode (practice account vs live)
- Notifications (Telegram alerts)
- Risk guards (drawdown guard, drift detector)
- Research (Twitter/Reddit/RSS)
- Backtesting mode
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

SETTINGS_FILE = "trading/data/settings.json"


@dataclass
class FeatureToggles:
    """Master on/off switches for every feature."""

    # ─── CORE ───
    scanning_enabled: bool = False         # Scan forex pairs for signals (OFF until you /resume)
    auto_trading_enabled: bool = True      # Place orders on OANDA practice when scanning is on
    demo_mode: bool = True                 # True = OANDA practice, False = LIVE money

    # ─── NOTIFICATIONS ───
    telegram_alerts: bool = True           # Send Telegram messages
    telegram_trade_alerts: bool = True     # Alert on trades placed
    telegram_scan_alerts: bool = True      # Alert on scan results
    telegram_win_loss_alerts: bool = True  # Alert on wins/losses

    # ─── AGENTS ───
    research_enabled: bool = True          # Run Twitter/Reddit/RSS research
    prediction_enabled: bool = True        # Run XGBoost + LLM predictions
    regime_detection_enabled: bool = False  # OFF for testing — blocks too much on weekends
    calibration_enabled: bool = False      # OFF for testing — needs 20+ trades to calibrate

    # ─── RISK GUARDS ───
    drawdown_guard_enabled: bool = True    # Block trading on max drawdown
    drift_detector_enabled: bool = False   # OFF for testing — enable after 20+ trades
    portfolio_manager_enabled: bool = False # OFF for testing — enable after data collected
    slippage_model_enabled: bool = False   # OFF for testing — enable for live trading
    data_quality_check: bool = False       # OFF for testing — weekend data triggers false alarms

    # ─── TRADING PARAMS ───
    max_trades_per_cycle: int = 3          # Max new trades per scan cycle
    max_pairs_to_scan: int = 28            # How many forex pairs to scan
    scan_interval_seconds: int = 300       # Time between scans (5 min default)

    # ─── MODES ───
    backtest_mode: bool = False            # Run backtester instead of live
    paper_trading: bool = True             # Log trades but don't execute
    runtime_profile: str = "paper"         # dev | paper | practice | live


class Settings:
    """
    Persistent settings manager.

    Load, save, and modify feature toggles.
    All changes are saved to disk immediately.
    """

    def __init__(self, settings_file: str = SETTINGS_FILE):
        self.settings_file = settings_file
        self.toggles = FeatureToggles()
        self._load()

    def _load(self):
        """Load settings from disk."""
        try:
            with open(self.settings_file) as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.toggles, key):
                        setattr(self.toggles, key, value)
            logger.info("Settings loaded from disk")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No settings file found — using defaults")

    def save(self):
        """Save current settings to disk."""
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        with open(self.settings_file, "w") as f:
            json.dump(asdict(self.toggles), f, indent=2)

    def set(self, key: str, value) -> bool:
        """Set a single setting. Returns True if valid key."""
        if hasattr(self.toggles, key):
            setattr(self.toggles, key, value)
            self.save()
            logger.info(f"Setting changed: {key} = {value}")
            return True
        return False

    def get(self, key: str, default=None):
        """Get a single setting value."""
        return getattr(self.toggles, key, default)

    def enable(self, feature: str) -> bool:
        """Turn a feature ON."""
        return self.set(feature, True)

    def disable(self, feature: str) -> bool:
        """Turn a feature OFF."""
        return self.set(feature, False)

    def go_live(self):
        """Switch from demo to LIVE trading. Requires explicit call."""
        self.toggles.demo_mode = False
        self.toggles.auto_trading_enabled = True
        self.toggles.paper_trading = False
        self.save()
        logger.warning("SWITCHED TO LIVE TRADING MODE")

    def go_demo(self):
        """Switch back to demo/practice mode."""
        self.toggles.demo_mode = True
        self.toggles.paper_trading = True
        self.toggles.auto_trading_enabled = False
        self.save()
        logger.info("Switched to demo mode")

    def get_status(self) -> str:
        """Get a human-readable status of all settings."""
        t = self.toggles
        mode = "LIVE" if not t.demo_mode else "DEMO"
        trading = "ON" if t.auto_trading_enabled else "OFF"

        lines = [
            "╔══════════════════════════════════════╗",
            "║        SYSTEM SETTINGS               ║",
            "╚══════════════════════════════════════╝",
            "",
            f"  Mode:           {mode} {'⚠️ REAL MONEY' if not t.demo_mode else '(practice)'}",
            f"  Auto-Trading:   {trading}",
            f"  Paper Trading:  {'ON' if t.paper_trading else 'OFF'}",
            "",
            "  ── Features ──",
            f"  Scanning:       {'ON' if t.scanning_enabled else 'OFF'}",
            f"  Research:       {'ON' if t.research_enabled else 'OFF'}",
            f"  Predictions:    {'ON' if t.prediction_enabled else 'OFF'}",
            f"  Regime Detect:  {'ON' if t.regime_detection_enabled else 'OFF'}",
            f"  Calibration:    {'ON' if t.calibration_enabled else 'OFF'}",
            "",
            "  ── Risk Guards ──",
            f"  Drawdown Guard: {'ON' if t.drawdown_guard_enabled else 'OFF'}",
            f"  Drift Detector: {'ON' if t.drift_detector_enabled else 'OFF'}",
            f"  Portfolio Mgr:  {'ON' if t.portfolio_manager_enabled else 'OFF'}",
            f"  Slippage Model: {'ON' if t.slippage_model_enabled else 'OFF'}",
            f"  Data Quality:   {'ON' if t.data_quality_check else 'OFF'}",
            "",
            "  ── Alerts ──",
            f"  Telegram:       {'ON' if t.telegram_alerts else 'OFF'}",
            f"  Trade Alerts:   {'ON' if t.telegram_trade_alerts else 'OFF'}",
            f"  Scan Alerts:    {'ON' if t.telegram_scan_alerts else 'OFF'}",
            f"  Win/Loss:       {'ON' if t.telegram_win_loss_alerts else 'OFF'}",
            "",
            "  ── Params ──",
            f"  Max Trades/Cycle: {t.max_trades_per_cycle}",
            f"  Pairs to Scan:    {t.max_pairs_to_scan}",
            f"  Scan Interval:    {t.scan_interval_seconds}s",
            "",
        ]
        return "\n".join(lines)

    def apply_profile(self, profile: str):
        """
        Apply a runtime profile that sets multiple toggles at once.

        Profiles:
            dev     — All guards off, paper trading, fast iteration
            paper   — Paper trading, drawdown guard on, most guards off
            practice — OANDA practice account, all guards on, real orders
            live    — REAL MONEY, all guards on, conservative settings
        """
        profiles = {
            "dev": {
                "scanning_enabled": False,
                "auto_trading_enabled": False,
                "demo_mode": True,
                "paper_trading": True,
                "drawdown_guard_enabled": False,
                "drift_detector_enabled": False,
                "portfolio_manager_enabled": False,
                "slippage_model_enabled": False,
                "data_quality_check": False,
                "regime_detection_enabled": False,
                "calibration_enabled": False,
                "runtime_profile": "dev",
            },
            "paper": {
                "scanning_enabled": False,
                "auto_trading_enabled": True,
                "demo_mode": True,
                "paper_trading": True,
                "drawdown_guard_enabled": True,
                "drift_detector_enabled": False,
                "portfolio_manager_enabled": False,
                "slippage_model_enabled": False,
                "data_quality_check": False,
                "regime_detection_enabled": False,
                "calibration_enabled": False,
                "runtime_profile": "paper",
            },
            "practice": {
                "scanning_enabled": True,
                "auto_trading_enabled": True,
                "demo_mode": True,
                "paper_trading": False,
                "drawdown_guard_enabled": True,
                "drift_detector_enabled": True,
                "portfolio_manager_enabled": True,
                "slippage_model_enabled": True,
                "data_quality_check": True,
                "regime_detection_enabled": True,
                "calibration_enabled": True,
                "runtime_profile": "practice",
            },
            "live": {
                "scanning_enabled": True,
                "auto_trading_enabled": True,
                "demo_mode": False,
                "paper_trading": False,
                "drawdown_guard_enabled": True,
                "drift_detector_enabled": True,
                "portfolio_manager_enabled": True,
                "slippage_model_enabled": True,
                "data_quality_check": True,
                "regime_detection_enabled": True,
                "calibration_enabled": True,
                "max_trades_per_cycle": 2,
                "runtime_profile": "live",
            },
        }

        if profile not in profiles:
            logger.warning(f"Unknown profile: {profile}")
            return False

        for key, value in profiles[profile].items():
            setattr(self.toggles, key, value)
        self.save()
        logger.info(f"Applied runtime profile: {profile}")
        return True

    def to_dict(self) -> dict:
        """Export all settings as dict."""
        return asdict(self.toggles)

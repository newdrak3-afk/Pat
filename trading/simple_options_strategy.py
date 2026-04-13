# trading/simple_options_strategy.py
"""
Simple Options Strategy — Direct trade execution with minimal dependencies.

Designed as a fallback when the full scanner produces no signals.
Uses only:
  - Daily candles for trend direction (20-day SMA)
  - Contracts endpoint for chain (no snapshots API required)
  - close_price from contract metadata as synthetic pricing

Entry logic:
  - Price > 20-day SMA → buy call
  - Price < 20-day SMA → buy put
  - Selects near-ATM contract, 7-21 DTE
  - Places limit order at ask (aggressive fill)

This intentionally skips confidence scoring, news, volume checks, etc.
Goal: prove the pipeline can actually PLACE AN OPTION ORDER.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Symbols to try in order of preference
SIMPLE_SYMBOLS = ["SPY", "QQQ", "IWM"]


class SimpleOptionsStrategy:
    """
    Dead-simple directional options strategy.

    Only requires:
      - Alpaca broker connected
      - Options trading enabled on the account
      - SPY/QQQ/IWM contracts available

    Does NOT require:
      - Snapshots API (paid plan)
      - Live bid/ask quotes
      - Complex indicator alignment
    """

    def __init__(self, broker, notifier=None):
        self.broker = broker
        self.notifier = notifier

    def get_trend(self, symbol: str) -> tuple[str, float, float]:
        """
        Get trend direction for a symbol using daily SMA20.

        Returns:
            (direction, current_price, sma20)
            direction: "buy" (calls) or "sell" (puts)
        """
        candles = self.broker.get_candles(symbol, "1Day", 25)
        if len(candles) < 20:
            raise ValueError(f"Not enough daily candles for {symbol}: got {len(candles)}")

        closes = [c["close"] for c in candles]
        current = closes[-1]
        sma20 = sum(closes[-20:]) / 20

        direction = "buy" if current > sma20 else "sell"
        logger.info(
            f"SIMPLE TREND {symbol}: close={current:.2f} sma20={sma20:.2f} "
            f"→ {'CALLS' if direction == 'buy' else 'PUTS'}"
        )
        return direction, current, sma20

    def get_contracts(
        self,
        symbol: str,
        option_type: str,
        current_price: float,
        min_dte: int = 7,
        max_dte: int = 21,
    ) -> list[dict]:
        """
        Fetch option contracts directly from Alpaca contracts endpoint.
        Does NOT use snapshots API.

        Returns list of raw contract dicts sorted by DTE.
        """
        today = datetime.now(timezone.utc).date()
        exp_gte = (today + timedelta(days=min_dte)).strftime("%Y-%m-%d")
        exp_lte = (today + timedelta(days=max_dte)).strftime("%Y-%m-%d")

        # Strike range: 95-105% of current price (near-ATM)
        strike_lo = current_price * 0.95
        strike_hi = current_price * 1.05

        params = {
            "underlying_symbols": symbol,
            "type": option_type,
            "status": "active",
            "expiration_date_gte": exp_gte,
            "expiration_date_lte": exp_lte,
            "strike_price_gte": f"{strike_lo:.2f}",
            "strike_price_lte": f"{strike_hi:.2f}",
            "limit": 50,
        }

        resp = requests.get(
            f"{self.broker.base_url}/v2/options/contracts",
            headers=self.broker._headers,
            params=params,
            timeout=15,
        )

        if not resp.ok:
            raise RuntimeError(
                f"Contracts API {resp.status_code}: {resp.text[:300]}"
            )

        contracts = resp.json().get("option_contracts", [])
        logger.info(
            f"SIMPLE CHAIN {symbol} {option_type}: {len(contracts)} contracts "
            f"({exp_gte} → {exp_lte}, strike {strike_lo:.0f}-{strike_hi:.0f})"
        )
        return contracts

    def pick_contract(
        self,
        contracts: list[dict],
        current_price: float,
        option_type: str,
    ) -> Optional[dict]:
        """
        Pick the best near-ATM contract.

        Preference:
        1. Strike closest to current price (ATM)
        2. DTE between 7-14 (sweet spot for premium)
        3. Non-zero close_price (has some pricing data)
        """
        if not contracts:
            return None

        def score(c):
            strike = float(c.get("strike_price") or 0)
            close_price = float(c.get("close_price") or 0)
            exp_str = c.get("expiration_date", "")
            try:
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp - datetime.now(timezone.utc).date()).days
            except ValueError:
                dte = 99

            # Prefer: has price data, DTE in sweet spot, close to ATM
            has_price = 1 if close_price > 0 else 0
            dte_score = 1 if 7 <= dte <= 14 else (0.5 if dte <= 21 else 0)
            atm_dist = abs(strike - current_price) / current_price  # lower = better

            return (has_price * 10) + (dte_score * 5) - (atm_dist * 100)

        ranked = sorted(contracts, key=score, reverse=True)
        best = ranked[0]

        strike = float(best.get("strike_price") or 0)
        close_price = float(best.get("close_price") or 0)
        exp_str = best.get("expiration_date", "")
        try:
            exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp - datetime.now(timezone.utc).date()).days
        except ValueError:
            dte = 0

        logger.info(
            f"SELECTED: {best.get('symbol')} | "
            f"strike={strike} | DTE={dte} | close_price={close_price:.2f}"
        )
        return best

    def place_trade(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        dry_run: bool = False,
    ) -> dict:
        """
        Execute a simple directional options trade.

        Returns a result dict with:
          success, symbol, contract_symbol, side, option_type,
          strike, expiration, dte, price, message
        """
        result = {
            "success": False,
            "symbol": symbol,
            "direction": direction,
            "message": "",
        }

        option_type = "call" if direction == "buy" else "put"

        # ── Step 1: Fetch contracts ──
        try:
            contracts = self.get_contracts(symbol, option_type, current_price)
        except RuntimeError as e:
            result["message"] = f"Contracts API failed: {e}"
            logger.warning(f"SIMPLE STRATEGY: {result['message']}")
            return result

        if not contracts:
            # Widen strike range and retry once
            logger.info(f"No ATM contracts found — widening strike range to 90-110%")
            try:
                contracts = self.get_contracts(
                    symbol, option_type, current_price,
                    min_dte=3, max_dte=30,
                )
            except RuntimeError as e:
                result["message"] = f"Contracts API failed (wide): {e}"
                return result

        if not contracts:
            result["message"] = f"No {option_type} contracts found for {symbol} (7-21 DTE, ATM±5%)"
            return result

        # ── Step 2: Pick best contract ──
        contract = self.pick_contract(contracts, current_price, option_type)
        if not contract:
            result["message"] = "Contract selection returned None"
            return result

        contract_symbol = contract.get("symbol", "")
        strike = float(contract.get("strike_price") or 0)
        expiration = contract.get("expiration_date", "")
        close_price = float(contract.get("close_price") or 0)

        try:
            exp = datetime.strptime(expiration, "%Y-%m-%d").date()
            dte = (exp - datetime.now(timezone.utc).date()).days
        except ValueError:
            dte = 0

        if not contract_symbol:
            result["message"] = "Contract has no symbol"
            return result

        # ── Step 3: Determine price ──
        # Try to get live quote first, fall back to close_price
        limit_price = 0.0
        try:
            quote_resp = requests.get(
                f"{self.broker.data_url}/v1beta1/options/quotes/latest",
                headers=self.broker._headers,
                params={"symbols": contract_symbol, "feed": "indicative"},
                timeout=10,
            )
            if quote_resp.ok:
                q = quote_resp.json().get("quotes", {}).get(contract_symbol, {})
                bid = float(q.get("bp", 0) or 0)
                ask = float(q.get("ap", 0) or 0)
                if ask > 0:
                    limit_price = ask  # Use ask for aggressive fill
                    logger.info(f"Live quote: bid={bid:.2f} ask={ask:.2f}")
        except Exception:
            pass

        # Fall back to close_price with 10% markup (we're buying, be willing to pay a bit more)
        if limit_price <= 0 and close_price > 0:
            limit_price = round(close_price * 1.05, 2)
            logger.info(f"Using synthetic price: close={close_price:.2f} → limit={limit_price:.2f}")

        if limit_price <= 0:
            result["message"] = f"No pricing data for {contract_symbol} (close_price={close_price})"
            return result

        max_loss = limit_price * 100  # 1 contract = 100 shares
        result.update({
            "contract_symbol": contract_symbol,
            "option_type": option_type,
            "strike": strike,
            "expiration": expiration,
            "dte": dte,
            "price": limit_price,
            "max_loss": max_loss,
        })

        # ── Step 4: Safety check ──
        if max_loss > 2000:
            result["message"] = (
                f"Premium too high: ${limit_price:.2f} × 100 = ${max_loss:.0f} "
                f"(limit $2000)"
            )
            return result

        if dry_run:
            result["success"] = True
            result["message"] = (
                f"DRY RUN — would buy {contract_symbol} @ ${limit_price:.2f} "
                f"(max loss ${max_loss:.0f})"
            )
            return result

        # ── Step 5: Place order ──
        logger.info(
            f"PLACING: {contract_symbol} BUY 1 @ ${limit_price:.2f} limit "
            f"(max loss ${max_loss:.0f})"
        )

        order_resp = requests.post(
            f"{self.broker.base_url}/v2/orders",
            headers=self.broker._headers,
            json={
                "symbol": contract_symbol,
                "qty": "1",
                "side": "buy",
                "type": "limit",
                "time_in_force": "day",
                "limit_price": str(limit_price),
            },
            timeout=15,
        )

        if order_resp.ok:
            order_data = order_resp.json()
            result["success"] = True
            result["order_id"] = order_data.get("id", "")
            result["order_status"] = order_data.get("status", "")
            result["message"] = (
                f"ORDER PLACED: {contract_symbol} @ ${limit_price:.2f} | "
                f"status={order_data.get('status')} | id={order_data.get('id', '')[:8]}"
            )
            logger.info(f"SIMPLE STRATEGY: {result['message']}")
        else:
            result["message"] = (
                f"Order rejected ({order_resp.status_code}): {order_resp.text[:300]}"
            )
            logger.warning(f"SIMPLE STRATEGY: {result['message']}")

        return result

    def run(
        self,
        symbols: list[str] = None,
        dry_run: bool = False,
        max_loss_per_trade: float = 1000.0,
    ) -> list[dict]:
        """
        Try to place one directional trade on each symbol.

        Returns list of result dicts.
        """
        if symbols is None:
            symbols = SIMPLE_SYMBOLS

        results = []
        for symbol in symbols:
            try:
                direction, current_price, sma20 = self.get_trend(symbol)
                result = self.place_trade(symbol, direction, current_price, dry_run=dry_run)
                result["current_price"] = current_price
                result["sma20"] = sma20
                results.append(result)

                if result["success"]:
                    logger.info(f"SIMPLE STRATEGY SUCCESS: {symbol}")
                    break  # One trade is enough — stop after first success
                else:
                    logger.info(f"SIMPLE STRATEGY {symbol} skipped: {result['message']}")

            except Exception as e:
                logger.warning(f"SIMPLE STRATEGY error for {symbol}: {e}", exc_info=True)
                results.append({
                    "success": False,
                    "symbol": symbol,
                    "message": str(e),
                })

        return results

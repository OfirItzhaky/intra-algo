from __future__ import annotations

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

import requests


logger = logging.getLogger("company_calendar")
logger.setLevel(logging.INFO)


class CompanyCalendarError(Exception):
    pass


def _first_nonempty(*vals):
    for v in vals:
        if v:
            return v
    return None


def _safe_get(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def get_upcoming_report_info(symbol: str, config: dict, days_ahead: int = 14) -> Dict[str, Any]:
    """
    Returns next-report window + tiny company summary. Uses whichever key exists in CONFIG.
    Order: Polygon -> Finnhub -> FMP -> mock.
    Raises CompanyCalendarError on validation or provider errors.
    """
    if not symbol or not isinstance(symbol, str):
        raise CompanyCalendarError("symbol is required and must be a non-empty string")

    symbol = symbol.upper().strip()
    end_date = (datetime.utcnow() + timedelta(days=days_ahead)).date().isoformat()

    # Collect possible keys from CONFIG (support different naming styles)
    polygon_key = _first_nonempty(
        config.get("POLYGON_API_KEY"),
        config.get("polygon_api_key"),
        config.get("polygon_key"),
    )
    finnhub_key = _first_nonempty(
        config.get("FINNHUB_API_KEY"),
        config.get("finnhub_api_key"),
        config.get("finnhub_key"),
    )
    fmp_key = _first_nonempty(
        config.get("FMP_API_KEY"),
        config.get("fmp_api_key"),
        config.get("fmp_key"),
    )

    # --- Provider 1: Polygon (earnings & company) ---
    if polygon_key:
        try:
            logger.info(f"[report-bias] Using Polygon for {symbol}")
            # Company profile
            prof_resp = requests.get(
                f"https://api.polygon.io/v3/reference/tickers/{symbol}",
                params={"apiKey": polygon_key},
                timeout=10,
            )
            if prof_resp.status_code == 403:
                logger.warning("[report-bias] Polygon access denied; falling back")
            else:
                prof_resp.raise_for_status()
                prof = prof_resp.json().get("results") or {}
                company_summary = (prof.get("name") or symbol) + " — " + (
                    prof.get("sic_description") or prof.get("description") or "No description"
                )
                # Earnings events (beta namespace). Fail soft if not available
                try:
                    earn = requests.get(
                        "https://api.polygon.io/vX/reference/events",
                        params={
                            "apiKey": polygon_key,
                            "ticker": symbol,
                            "types": "earnings",
                            "limit": 50,
                        },
                        timeout=10,
                    )
                    if earn.ok:
                        items = earn.json().get("results") or []
                        upcoming = None
                        today = datetime.utcnow().date()
                        for it in items:
                            dt = _safe_get(it, "start_date") or _safe_get(it, "date") or _safe_get(it, "earnings_date")
                            if dt:
                                try:
                                    ddt = datetime.fromisoformat(dt[:10]).date()
                                except Exception:
                                    continue
                                if today <= ddt <= datetime.fromisoformat(end_date).date():
                                    upcoming = ddt.isoformat()
                                    break
                        return {
                            "symbol": symbol,
                            "has_upcoming_report": upcoming is not None,
                            "report_date": upcoming,
                            "company_summary": company_summary,
                            "provider_used": "polygon",
                        }
                except Exception as e:
                    logger.warning(f"[report-bias] Polygon earnings look-up failed: {e}")

        except Exception as e:
            logger.error(f"[report-bias] Polygon error: {e}")

    # --- Provider 2: Finnhub ---
    if finnhub_key:
        try:
            logger.info(f"[report-bias] Using Finnhub for {symbol}")
            # Company profile
            prof = requests.get(
                "https://finnhub.io/api/v1/stock/profile2",
                params={"symbol": symbol, "token": finnhub_key},
                timeout=10,
            )
            prof.raise_for_status()
            pjs = prof.json() or {}
            company_summary = f"{pjs.get('name') or symbol} — {pjs.get('finnhubIndustry') or 'No sector info'}"

            # Earnings calendar window
            start = datetime.utcnow().date().isoformat()
            earn = requests.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={"from": start, "to": end_date, "symbol": symbol, "token": finnhub_key},
                timeout=10,
            )
            earn.raise_for_status()
            ejs = earn.json() or {}
            rows = ejs.get("earningsCalendar") or []
            upcoming = None
            for r in rows:
                d = r.get("date") or r.get("period")
                if not d:
                    continue
                try:
                    ddt = datetime.fromisoformat(d[:10]).date()
                except Exception:
                    continue
                if ddt >= datetime.utcnow().date():
                    upcoming = ddt.isoformat()
                    break

            return {
                "symbol": symbol,
                "has_upcoming_report": upcoming is not None,
                "report_date": upcoming,
                "company_summary": company_summary,
                "provider_used": "finnhub",
            }

        except Exception as e:
            logger.error(f"[report-bias] Finnhub error: {e}")

    # --- Provider 3: FMP (Financial Modeling Prep) ---
    if fmp_key:
        try:
            logger.info(f"[report-bias] Using FMP for {symbol}")
            # Company profile
            prof = requests.get(
                f"https://financialmodelingprep.com/api/v3/profile/{symbol}",
                params={"apikey": fmp_key},
                timeout=10,
            )
            prof.raise_for_status()
            pjs = (prof.json() or [])
            p0 = pjs[0] if pjs else {}
            company_summary = f"{p0.get('companyName') or symbol} — {p0.get('sector') or 'No sector info'}"

            # Earnings calendar (next)
            cal = requests.get(
                "https://financialmodelingprep.com/api/v3/earning_calendar",
                params={
                    "symbol": symbol,
                    "from": datetime.utcnow().date().isoformat(),
                    "to": end_date,
                    "apikey": fmp_key,
                },
                timeout=10,
            )
            cal.raise_for_status()
            items = cal.json() or []
            upcoming = None
            for it in items:
                d = it.get("date")
                if not d:
                    continue
                try:
                    ddt = datetime.fromisoformat(d[:10]).date()
                except Exception:
                    continue
                if ddt >= datetime.utcnow().date():
                    upcoming = ddt.isoformat()
                    break

            return {
                "symbol": symbol,
                "has_upcoming_report": upcoming is not None,
                "report_date": upcoming,
                "company_summary": company_summary,
                "provider_used": "fmp",
            }

        except Exception as e:
            logger.error(f"[report-bias] FMP error: {e}")

    # --- Fallback: mock, still useful for testing route ---
    logger.info(f"[report-bias] No provider keys found; returning mock for {symbol}")
    return {
        "symbol": symbol,
        "has_upcoming_report": False,
        "report_date": None,
        "company_summary": f"{symbol} — summary unavailable (mock, no API key configured)",
        "provider_used": "mock",
    }


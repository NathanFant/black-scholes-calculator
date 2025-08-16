#!/usr/bin/env python3
# pip install yfinance pandas numpy

from __future__ import annotations
import argparse
import datetime as dt
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf

OptionType = Literal["call", "c", "put", "p"]


def binomial_option(
    S: float,
    K: float,
    T: float,  # years
    r: float,  # risk-free (annual, decimal)
    q: float,  # dividend yield (annual, decimal)
    sigma: float,  # vol (annual, decimal)
    steps: int,
    option_type: OptionType,
) -> float:
    if T <= 0.0:
        if option_type in ("call", "c"):
            return float(max(0.0, S - K))
        if option_type in ("put", "p"):
            return float(max(0.0, K - S))
        raise ValueError("option_type must be 'call'/'c' or 'put'/'p'")

    sigma = float(np.clip(sigma, 1e-8, 10.0))
    steps = max(int(steps), 1)

    dtau = T / steps
    u = float(np.exp(sigma * np.sqrt(dtau)))
    d = 1.0 / u
    disc = float(np.exp(-r * dtau))
    p = (np.exp((r - q) * dtau) - d) / (u - d)
    p = float(np.clip(p, 0.0, 1.0))

    j = np.arange(steps + 1, dtype=float)
    ST = S * (u**j) * (d ** (steps - j))

    if option_type in ("call", "c"):
        V = np.maximum(0.0, ST - K)
    elif option_type in ("put", "p"):
        V = np.maximum(0.0, K - ST)
    else:
        raise ValueError("option_type must be 'call'/'c' or 'put'/'p'")

    for n in range(steps - 1, -1, -1):
        ST = ST[: n + 1] / u
        V = disc * (p * V[1 : n + 2] + (1.0 - p) * V[: n + 1])
        exercise = (
            np.maximum(0.0, ST - K)
            if option_type in ("call", "c")
            else np.maximum(0.0, K - ST)
        )
        V = np.maximum(V, exercise)

    return float(V[0])


def annualize_t(expiry_date: dt.date) -> float:
    today = dt.datetime.utcnow().date()
    if expiry_date <= today:
        return 0.0
    return (expiry_date - today).days / 365.0


def _yf_last_percent(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        val = tk.fast_info.last_price
        if val is None or not np.isfinite(val):
            hist = tk.history(period="1d")["Close"]
            if hist.empty:
                return None
            val = float(hist.iloc[-1])
        return float(val) / 100.0
    except Exception:
        return None


def get_risk_free_rate_for_T(T: float, fallback: float = 0.05) -> float:
    # <=13w: ^IRX, <=5y: ^FVX, else: ^TNX
    if T <= 0.25:
        r = _yf_last_percent("^IRX")
        return r if r is not None else fallback
    if T <= 5.0:
        r = _yf_last_percent("^FVX")
        return r if r is not None else fallback
    r = _yf_last_percent("^TNX")
    return r if r is not None else fallback


def compute_fair_option_price(
    ticker: str,
    r: Optional[float] = None,
    q: float = 0.0,
    steps: int = 400,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    tk = yf.Ticker(ticker.upper())
    S = tk.fast_info.last_price
    if S is None or not np.isfinite(S):
        hist = tk.history(period="1d")["Close"]
        if hist.empty:
            raise ValueError(f"Could not fetch spot for {ticker}")
        S = float(hist.iloc[-1])

    exps = tk.options
    if not exps:
        raise ValueError(f"No options available for {ticker}")

    rows = []
    for exp in exps:
        expiry = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        T = annualize_t(expiry)
        chain = tk.option_chain(exp)
        r_T = r if r is not None else get_risk_free_rate_for_T(T)

        for typ, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            strikes = df["strike"].to_numpy(float)
            iv = df["impliedVolatility"].to_numpy(float)  # 2.5 == 250%
            bid = df["bid"].to_numpy(float)
            ask = df["ask"].to_numpy(float)
            last = df["lastPrice"].to_numpy(float)
            mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)

            for i in range(len(df)):
                sigma = float(np.clip(iv[i], 1e-8, 10.0))
                fv = binomial_option(S, float(strikes[i]), T, r_T, q, sigma, steps, typ)
                rows.append(
                    {
                        "expiry": exp,
                        "type": "C" if typ == "call" else "P",
                        "strike": round(float(strikes[i]), 4),
                        "IV(%)": round(sigma * 100.0, 2),
                        "mid": round(float(mid[i]), 4) if np.isfinite(mid[i]) else None,
                        "fair_value": round(fv, 4),
                        "r_used": round(r_T, 6),
                    }
                )

    out = pd.DataFrame(rows).sort_values(["expiry", "type", "strike"])
    print(
        "\nUnderlying {} spot ≈ {:.4f} | q={} | steps={}\n".format(
            ticker.upper(), S, f"{q:.3%}", steps
        )
    )
    for exp, df_exp in out.groupby("expiry"):
        r_unique = sorted(set(df_exp["r_used"]))
        r_str = ", ".join(f"{x:.3%}" for x in r_unique)
        print(f"=== {exp} ===  (r used: {r_str})")
        print(df_exp.drop(columns=["r_used"]).to_string(index=False))
        print()

    if out_csv:
        out.to_csv(out_csv, index=False)
        print(f"Output saved to {out_csv}")

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="American option fair values by expiry (CRR binomial)."
    )
    ap.add_argument("ticker", nargs="?", default="SPY", help="Ticker (default: SPY)")
    ap.add_argument(
        "--r",
        type=float,
        default=None,
        help="Risk-free rate (annualized, decimal). If omitted, auto-fetch per expiry.",
    )
    ap.add_argument(
        "--q", type=float, default=0.0, help="Dividend yield (annualized, decimal)"
    )
    ap.add_argument(
        "--steps", type=int, default=400, help="Binomial steps (default: 400)"
    )
    ap.add_argument("--csv", type=str, default=None, help="CSV output path")
    args = ap.parse_args()

    # default CSV name if not provided
    today = dt.date.today().strftime("%m_%d_%Y")
    csv_path = args.csv or f"{args.ticker.lower()}_{today}.csv"

    df = compute_fair_option_price(
        args.ticker, r=args.r, q=args.q, steps=args.steps, out_csv=csv_path
    )

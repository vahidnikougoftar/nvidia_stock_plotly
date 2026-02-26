"""
Robinhood/TradingView-ish dark-mode candlestick + volume with:
- OHLCV strip on top that updates on hover (per candle)
- Bottom interval dropdown: 1m / 5m / 30m / 1h (resampled from 1m)
- Vertical-axis “press + hold + drag up/down” to zoom Y range (custom JS)
- Auto-scale toggle (checkbox) like the screenshot

Data source: Yahoo Finance via yfinance (free).
Note: 1-minute data is usually limited (often ~5–7 trading days). Use --period 5d, etc.

Install:
  pip install yfinance pandas plotly

Run:
  python robinhood_like_chart.py --ticker NVDA --period 5d --out nvda_robinhood_like.html
Open the HTML in your browser.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio

from datetime import datetime, timezone, timedelta

@dataclass(frozen=True)
class IntervalSpec:
    label: str
    rule: str


INTERVALS: Dict[str, IntervalSpec] = {
    "1m": IntervalSpec("1 minute", "1min"),
    "5m": IntervalSpec("5 minute", "5min"),
    "30m": IntervalSpec("30 minute", "30min"),
    "1h": IntervalSpec("1 hour", "1H"),
}


def download_1m_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1m",
        auto_adjust=False,
        progress=False,
        threads=True,
        prepost=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} (period={period}, interval=1m). "
            "Try a smaller period like 5d, or a larger interval."
        )

    # Normalize possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df["Volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _vol_colors(dfi: pd.DataFrame) -> list:
    up = dfi["Close"] >= dfi["Open"]
    # green / orange-red like your screenshot
    return ["rgba(0,200,0,0.55)" if u else "rgba(255,90,0,0.55)" for u in up]


def build_figure(dfs: Dict[str, pd.DataFrame], ticker: str, default_interval: str) -> go.Figure:
    fig = go.Figure()
    interval_keys = list(INTERVALS.keys())

    # Create 2 traces per interval: candles + volume
    # Visible only for default interval at start
    for k in interval_keys:
        dfi = dfs[k]
        x = dfi.index

        custom = pd.DataFrame(
            {
                "o": dfi["Open"].round(2),
                "h": dfi["High"].round(2),
                "l": dfi["Low"].round(2),
                "c": dfi["Close"].round(2),
                "v": dfi["Volume"],
            }
        ).values

        show = (k == default_interval)

        fig.add_trace(
            go.Candlestick(
                x=x,
                open=dfi["Open"],
                high=dfi["High"],
                low=dfi["Low"],
                close=dfi["Close"],
                name=f"Price {k}",
                visible=show,
                customdata=custom,
                hovertemplate=(
                    "<b>%{x|%b %d %H:%M}</b><br>"
                    "O %{customdata[0]:.2f} &nbsp; "
                    "H %{customdata[1]:.2f} &nbsp; "
                    "L %{customdata[2]:.2f} &nbsp; "
                    "C %{customdata[3]:.2f}<br>"
                    "V %{customdata[4]:,}<extra></extra>"
                ),
                increasing_line_width=1,
                decreasing_line_width=1,
            )
        )

        fig.add_trace(
            go.Bar(
                x=x,
                y=dfi["Volume"],
                name=f"Volume {k}",
                visible=show,
                marker=dict(color=_vol_colors(dfi)),
                yaxis="y2",
                hoverinfo="skip",
                opacity=0.95,
            )
        )

    # Dark mode styling
    bg = "#0b1220"       # page
    panel = "#0f172a"    # chart panel
    grid = "rgba(255,255,255,0.08)"
    axis = "rgba(255,255,255,0.55)"
    text = "rgba(255,255,255,0.9)"

    fig.update_layout(
        margin=dict(l=30, r=70, t=10, b=30),
        paper_bgcolor=bg,
        plot_bgcolor=panel,
        font=dict(color=text, size=14),
        showlegend=False,

        xaxis=dict(
            type="date",
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color=axis),
            rangeslider=dict(visible=False),  # Robinhood-like: no rangeslider
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
            spikecolor="rgba(255,255,255,0.35)",
            spikethickness=1,
        ),
        yaxis=dict(
            side="right",
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            tickfont=dict(color=axis),
            tickformat=".2f",
            fixedrange=False,  # allow zoom/pan (we'll also do our own vertical drag)
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
            showgrid=False,
            visible=False,
            rangemode="tozero",
        ),
        hovermode="x",
        dragmode="pan",
    )

    # Make candles a bit chunkier
    fig.update_traces(selector=dict(type="candlestick"),
                      increasing_fillcolor="rgba(0,200,0,0.75)",
                      decreasing_fillcolor="rgba(255,90,0,0.75)",
                      increasing_line_color="rgba(0,200,0,1.0)",
                      decreasing_line_color="rgba(255,90,0,1.0)")

    return fig


def make_html(fig: go.Figure, ticker: str, default_interval: str,generated_utc: str) -> str:
    # Embed plotly as CDN for a smaller file
    plot_div = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="chart")

    # We need to know how many traces + which traces correspond to each interval
    # We created 2 traces per interval in the exact order of INTERVALS keys.
    interval_keys = list(INTERVALS.keys())
    trace_map = {}
    for i, k in enumerate(interval_keys):
        candle_idx = 2 * i
        vol_idx = 2 * i + 1
        trace_map[k] = [candle_idx, vol_idx]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{ticker} - Robinhood-like Candles</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #0f172a;
      --muted: rgba(255,255,255,0.65);
      --muted2: rgba(255,255,255,0.45);
      --text: rgba(255,255,255,0.92);
      --green: #00c800;
      --orange: #ff5a00;
      --chip: rgba(255,255,255,0.08);
      --chip2: rgba(255,255,255,0.12);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 10px 14px 14px 14px;
    }}

    /* Top OHLCV strip */
    .topbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      margin-bottom: 10px;
      user-select: none;
    }}
    .ticker {{
      font-weight: 700;
      letter-spacing: 0.5px;
      padding-right: 8px;
      border-right: 1px solid rgba(255,255,255,0.10);
    }}
    .ohlcv {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      font-variant-numeric: tabular-nums;
    }}
    .kv {{
      color: var(--muted);
    }}
    .kv b {{
      color: var(--text);
      font-weight: 700;
      margin-left: 4px;
    }}
    .kv .lbl {{
      color: var(--muted2);
      margin-right: 2px;
    }}

    /* Chart container */
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 6px 6px 2px 6px;
      position: relative;
    }}

    /* Bottom controls like screenshot */
    .bottombar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 6px 4px 6px;
      color: var(--muted);
      user-select: none;
    }}
    .leftControls {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .chip {{
      background: var(--chip);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      padding: 6px 10px;
      display: inline-flex;
      gap: 8px;
      align-items: center;
    }}
    select {{
      background: var(--chip2);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 10px;
      padding: 6px 10px;
      outline: none;
      cursor: pointer;
    }}
    .autoscale {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    input[type="checkbox"] {{
      width: 16px; height: 16px;
      accent-color: var(--text);
      cursor: pointer;
    }}
    .hint {{
      color: var(--muted2);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="ticker">{ticker}</div>

      <div class="ohlcv" id="ohlcv">
        <span class="kv"><span class="lbl">O</span><b id="o">—</b></span>
        <span class="kv"><span class="lbl">H</span><b id="h">—</b></span>
        <span class="kv"><span class="lbl">L</span><b id="l">—</b></span>
        <span class="kv"><span class="lbl">C</span><b id="c">—</b></span>
        <span class="kv"><span class="lbl">V</span><b id="v">—</b></span>
      </div>
        <div style="margin-left:auto; color: rgba(255,255,255,0.65); font-variant-numeric: tabular-nums;">
          Last updated: <b style="color: rgba(255,255,255,0.92);">{generated_utc}</b>
        </div>
    </div>

    <div class="panel">
      {plot_div}
      <div class="bottombar">
        <div class="leftControls">
          <div class="chip">
            <span>Interval:</span>
            <select id="intervalSelect">
              <option value="1m">1 minute</option>
              <option value="5m">5 minute</option>
              <option value="30m">30 minute</option>
              <option value="1h">1 hour</option>
            </select>
          </div>

          <div class="autoscale chip">
            <input type="checkbox" id="autoscale" checked />
            <label for="autoscale">Auto-scale</label>
          </div>
        </div>

        <div class="hint">Tip: press & hold in chart, drag ↑/↓ to zoom Y-axis</div>
      </div>
    </div>
  </div>

<script>
  // ---------- Config ----------
  const defaultInterval = {json.dumps(default_interval)};
  const traceMap = {json.dumps(trace_map)};
  const chartId = "chart";

  const oEl = document.getElementById("o");
  const hEl = document.getElementById("h");
  const lEl = document.getElementById("l");
  const cEl = document.getElementById("c");
  const vEl = document.getElementById("v");

  const intervalSelect = document.getElementById("intervalSelect");
  const autoscale = document.getElementById("autoscale");

  // Set default dropdown
  intervalSelect.value = defaultInterval;

  // ---------- Helpers ----------
  function fmt2(x) {{
    if (x === null || x === undefined || isNaN(x)) return "—";
    return Number(x).toFixed(2);
  }}

  function fmtVol(x) {{
    if (x === null || x === undefined || isNaN(x)) return "—";
    return Number(x).toLocaleString();
  }}

  function setOHLCV(o, h, l, c, v) {{
    oEl.textContent = fmt2(o);
    hEl.textContent = fmt2(h);
    lEl.textContent = fmt2(l);
    cEl.textContent = fmt2(c);
    vEl.textContent = fmtVol(v);
  }}

  function showInterval(intervalKey) {{
    // Build "visible" array length = total traces
    const gd = document.getElementById(chartId);
    const n = gd.data.length;
    const visible = new Array(n).fill(false);

    const idxs = traceMap[intervalKey];  // [candleIdx, volIdx]
    visible[idxs[0]] = true;
    visible[idxs[1]] = true;

    Plotly.restyle(gd, {{visible: visible}});

    // Force autoscale if enabled
    if (autoscale.checked) {{
      Plotly.relayout(gd, {{"yaxis.autorange": true}});
    }}
    // default X view to last 2 hours
    setTimeout(() => setLastTwoHoursView(intervalKey),0);

  }}

  // ---------- Hover -> update top OHLCV strip ----------
  const gd = document.getElementById(chartId);

  gd.on("plotly_hover", function(evt) {{
    // Candlestick points include customdata
    const pt = evt.points && evt.points[0];
    if (!pt) return;

    // We only want hover from candlestick trace, not volume bars
    if (pt.data && pt.data.type !== "candlestick") return;

    const cd = pt.customdata; // [o,h,l,c,v]
    if (!cd || cd.length < 5) return;
    setOHLCV(cd[0], cd[1], cd[2], cd[3], cd[4]);
  }});

  gd.on("plotly_unhover", function() {{
    // Keep last hovered values (Robinhood-ish). If you prefer reset, uncomment:
    // setOHLCV(null,null,null,null,null);
  }});

  // Initialize OHLCV with last candle of default interval
  function initOHLCV() {{
    const intervalKey = intervalSelect.value;
    const candleIdx = traceMap[intervalKey][0];
    const trace = gd.data[candleIdx];
    if (!trace || !trace.customdata || trace.customdata.length === 0) return;
    const last = trace.customdata[trace.customdata.length - 1];
    setOHLCV(last[0], last[1], last[2], last[3], last[4]);
  }}

  // ---------- Show last Two Hours View As Default ----------- //
   function setLastTwoHoursView(intervalKey) {{
  const candleIdx = traceMap[intervalKey][0];
  const trace = gd.data[candleIdx];
  if (!trace || !trace.x || trace.x.length < 2) return;

  const end = new Date(trace.x[trace.x.length - 1]).getTime();
  const start = end - (2 * 60 * 60 * 1000); // 2 hours

  Plotly.relayout(gd, {{
    "xaxis.autorange": false,
    "xaxis.range": [new Date(start).toISOString(), new Date(end).toISOString()]
  }});
}}


  // ---------- Bottom dropdown ----------
  intervalSelect.addEventListener("change", () => {{
    showInterval(intervalSelect.value);
    // After restyle, wait a tick, then update OHLCV from last candle
    setTimeout(initOHLCV, 0);
  }});

  // ---------- Auto-scale toggle ----------
  autoscale.addEventListener("change", () => {{
    if (autoscale.checked) {{
      Plotly.relayout(gd, {{"yaxis.autorange": true}});
    }}
  }});

  // ---------- Vertical axis press+hold+drag zoom ----------
  // Behavior: hold mouse inside plot area and drag up/down to zoom y-range around center.
  let dragging = false;
  let startY = 0;
  let startRange = null;

  function getYRange() {{
    const full = gd._fullLayout && gd._fullLayout.yaxis;
    if (!full || !full.range) return null;
    return [Number(full.range[0]), Number(full.range[1])];
  }}

  function setYRange(r0, r1) {{
    Plotly.relayout(gd, {{"yaxis.autorange": false, "yaxis.range": [r0, r1]}});
  }}

  // We attach to the plotly div, but only activate when mouse is down on the SVG drag layer
  gd.addEventListener("mousedown", (e) => {{
    // Only left button
    if (e.button !== 0) return;

    // If autoscale is checked, turning it off when user manually drags feels natural
    if (autoscale.checked) autoscale.checked = false;

    const r = getYRange();
    if (!r) return;
    dragging = true;
    startY = e.clientY;
    startRange = r;

    // Prevent text selection / unwanted behavior
    e.preventDefault();
  }});

  window.addEventListener("mousemove", (e) => {{
    if (!dragging || !startRange) return;

    const dy = e.clientY - startY;

    // Convert pixel movement into zoom factor.
    // dy > 0 (drag down) -> zoom out
    // dy < 0 (drag up) -> zoom in
    const k = Math.exp(dy * 0.004); // sensitivity
    const mid = (startRange[0] + startRange[1]) / 2.0;
    const half = (startRange[1] - startRange[0]) / 2.0;
    const newHalf = half * k;

    const r0 = mid - newHalf;
    const r1 = mid + newHalf;
    setYRange(r0, r1);
  }});

  window.addEventListener("mouseup", () => {{
    dragging = false;
    startRange = null;
  }});

  // ---------- First paint ----------
  // Make sure default interval is active and OHLCV shows last candle
  showInterval(defaultInterval);
  setTimeout(initOHLCV, 0);
</script>
</body>
</html>
"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--period", default="5d", help="For 1m data: try 1d, 5d, 7d-ish depending on Yahoo limits")
    ap.add_argument("--default_interval", default="5m", choices=list(INTERVALS.keys()))
    ap.add_argument("--out", default="robinhood_like.html")
    args = ap.parse_args()

    df1m = download_1m_ohlcv(args.ticker, args.period)

    dfs = {}
    for k, spec in INTERVALS.items():
        dfs[k] = df1m if k == "1m" else resample_ohlcv(df1m, spec.rule)

    fig = build_figure(dfs, args.ticker, args.default_interval)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = make_html(fig, args.ticker, args.default_interval,generated_utc)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved: {args.out}")
    print("Open it in your browser.")


if __name__ == "__main__":
    main()


import os, io, base64, json, time, uuid, pathlib, sys
import pandas as pd
import numpy as np

# Ensure backend root is on sys.path so imports work when running this file directly.
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from utils.data import normalize_columns, kpis
from models import registry

# persistence
from core import db

# async (optional)
try:
    import diskcache
    from dash.long_callback import DiskcacheLongCallbackManager

    cache = diskcache.Cache("./.cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)
except Exception:
    cache = None
    long_callback_manager = None

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="AI4CM — Cashflow Forecast",
    long_callback_manager=long_callback_manager,
    suppress_callback_exceptions=True,   # fixes "ID not found in layout"
)
server = app.server

# Ensure DB exists
db.init_db()
DATA_DIR = "data"
ART_DIR = "artifacts"
pathlib.Path(DATA_DIR).mkdir(exist_ok=True)
pathlib.Path(ART_DIR).mkdir(exist_ok=True)

# ── Colour palette ───────────────────────────────────────────────────
TEAL       = "#0d9488"
TEAL_LIGHT = "#ccfbf1"
SLATE_800  = "#1e293b"
SLATE_100  = "#f1f5f9"
SLATE_50   = "#f8fafc"
WHITE      = "#ffffff"
ROSE_600   = "#e11d48"


# ── Components ───────────────────────────────────────────────────────
def kpi_card(title, value, colour=TEAL):
    formatted = f"{value:,.0f}" if isinstance(value, (int, float)) and value is not None else "-"
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={
                "fontSize": "11px", "fontWeight": "700", "textTransform": "uppercase",
                "letterSpacing": "0.08em", "color": "#64748b", "marginBottom": "4px",
            }),
            html.H4(formatted, style={
                "fontWeight": "800", "color": colour, "marginBottom": "0",
                "fontVariantNumeric": "tabular-nums",
            }),
        ]),
        style={
            "borderRadius": "12px", "border": f"1.5px solid {SLATE_100}",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.04)",
        },
    )


def empty_runs_table():
    """Return an empty DataTable so callbacks can reference it at startup."""
    return dash_table.DataTable(
        id="runs-table",
        data=[],
        columns=[
            {"name": "Run", "id": "run_id"},
            {"name": "Model", "id": "model"},
            {"name": "H", "id": "horizon"},
            {"name": "Status", "id": "status"},
            {"name": "Started", "id": "started"},
            {"name": "Elapsed", "id": "elapsed_s"},
            {"name": "MAE", "id": "mae"},
            {"name": "RMSE", "id": "rmse"},
        ],
        page_size=8,
        row_selectable="single",
        style_table={"overflowY": "auto", "height": "340px"},
        style_header={
            "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
            "letterSpacing": "0.06em", "backgroundColor": SLATE_50,
            "borderBottom": f"2px solid {SLATE_100}", "color": "#64748b",
        },
        style_cell={
            "fontFamily": "'Inter', -apple-system, sans-serif",
            "fontSize": "13px", "padding": "8px 10px",
            "borderBottom": f"1px solid {SLATE_100}",
        },
        style_data_conditional=[
            {"if": {"filter_query": '{status} = "DONE"'}, "color": TEAL, "fontWeight": "600"},
            {"if": {"filter_query": '{status} = "FAILED"'}, "color": ROSE_600, "fontWeight": "600"},
            {"if": {"filter_query": '{status} = "RUNNING"'}, "color": "#d97706", "fontWeight": "600"},
        ],
    )


def runs_table(rows):
    records = []
    for r in rows:
        run_id, dataset_id, model, params, horizon, status, started_at, finished_at, metrics, error = r
        metrics = json.loads(metrics) if metrics else {}
        records.append({
            "run_id": run_id[:8],
            "model": model,
            "horizon": horizon,
            "status": status,
            "started": time.strftime("%m/%d %H:%M", time.localtime(started_at)),
            "elapsed_s": "-" if not finished_at else f"{finished_at - started_at:.1f}",
            "mae": None if not metrics else round(metrics.get("mae", None) or 0, 2),
            "rmse": None if not metrics else round(metrics.get("rmse", None) or 0, 2),
        })
    tbl = empty_runs_table()
    tbl.data = records
    return tbl


# ── Layout ───────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, style={"fontFamily": "'Inter', -apple-system, sans-serif"}, children=[

    html.Link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ),

    dbc.Row([
        # ── SIDEBAR ──────────────────────────────────────────────────
        dbc.Col([
            html.Div([
                html.H4("AI4CM", style={
                    "fontWeight": "800", "color": WHITE, "marginBottom": "2px",
                    "letterSpacing": "-0.02em",
                }),
                html.P("Cashflow Forecast Lab", style={
                    "fontSize": "13px", "color": "#94a3b8", "marginBottom": "20px",
                }),
            ], style={"padding": "20px 16px 0 16px"}),

            html.Div([
                html.Label("Upload Data", style={
                    "fontSize": "10px", "fontWeight": "700", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "#94a3b8", "marginBottom": "6px",
                }),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drop file or ", html.A("browse", style={"color": "#5eead4"})]),
                    style={
                        "width": "100%", "padding": "14px", "borderWidth": "1.5px",
                        "borderStyle": "dashed", "borderRadius": "10px",
                        "borderColor": "#334155", "textAlign": "center",
                        "color": "#94a3b8", "fontSize": "13px", "cursor": "pointer",
                        "backgroundColor": "rgba(255,255,255,0.03)",
                    },
                    multiple=False,
                ),
                html.Div(id="upload-status", style={"marginTop": "8px"}),
            ], style={"padding": "0 16px", "marginBottom": "16px"}),

            html.Div([
                html.Label("Dataset", style={
                    "fontSize": "10px", "fontWeight": "700", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "#94a3b8", "marginBottom": "4px",
                }),
                dcc.Dropdown(id="dataset-dd", options=[], placeholder="Select a dataset...",
                             style={"fontSize": "13px"}),
            ], style={"padding": "0 16px", "marginBottom": "14px"}),

            html.Div([
                html.Label("Model", style={
                    "fontSize": "10px", "fontWeight": "700", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "#94a3b8", "marginBottom": "4px",
                }),
                dcc.Dropdown(
                    id="model-dd",
                    options=[{"label": m, "value": m} for m in registry.list_models()],
                    value="A0 \u2022 Moving Average",
                    style={"fontSize": "13px"},
                ),
            ], style={"padding": "0 16px", "marginBottom": "14px"}),

            html.Div([
                html.Label("Horizon (days)", style={
                    "fontSize": "10px", "fontWeight": "700", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "#94a3b8", "marginBottom": "4px",
                }),
                dcc.Slider(
                    7, 180, 1, value=60, id="horizon",
                    marks={7: "7", 30: "30", 60: "60", 90: "90", 180: "180"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"padding": "0 16px", "marginBottom": "14px"}),

            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            html.Label("MA Window", style={"fontSize": "12px", "color": "#94a3b8"}),
                            dcc.Input(id="param-window", type="number", value=14,
                                      style={"width": "100%", "borderRadius": "8px"}),
                        ], md=6),
                        dbc.Col([
                            html.Label("Seasonal s", style={"fontSize": "12px", "color": "#94a3b8"}),
                            dcc.Input(id="param-s", type="number", value=7,
                                      style={"width": "100%", "borderRadius": "8px"}),
                        ], md=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("ARIMA (p, d, q)", style={"fontSize": "12px", "color": "#94a3b8"}),
                            html.Div([
                                dcc.Input(id="param-p", type="number", value=1,
                                          style={"width": "30%", "borderRadius": "8px", "marginRight": "4px"}),
                                dcc.Input(id="param-d", type="number", value=1,
                                          style={"width": "30%", "borderRadius": "8px", "marginRight": "4px"}),
                                dcc.Input(id="param-q", type="number", value=1,
                                          style={"width": "30%", "borderRadius": "8px"}),
                            ], style={"display": "flex"}),
                        ], md=12),
                    ]),
                ], title="Parameters"),
            ], start_collapsed=True, style={"margin": "0 16px 16px 16px"}),

            html.Div([
                dbc.Button(
                    "Run Experiment",
                    id="run-btn", color="primary",
                    style={
                        "width": "100%", "borderRadius": "10px", "fontWeight": "600",
                        "background": f"linear-gradient(135deg, {TEAL} 0%, #0f766e 100%)",
                        "border": "none", "padding": "10px",
                        "boxShadow": "0 2px 8px rgba(13,148,136,0.3)",
                    },
                ),
                html.Div(id="run-status", style={"marginTop": "8px"}),
            ], style={"padding": "0 16px", "marginBottom": "16px"}),

            html.Hr(style={"borderColor": "#334155", "margin": "0 16px"}),

            html.Div([
                html.P("Experiment History", style={
                    "fontSize": "10px", "fontWeight": "700", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "#94a3b8",
                    "marginTop": "14px", "marginBottom": "8px",
                }),
                html.Div(id="runs-div", children=[empty_runs_table()]),
            ], style={"padding": "0 16px"}),

            dcc.Interval(id="runs-refresh", interval=3000, n_intervals=0),
            dcc.Store(id="selected-run-id"),

        ], md=3, style={
            "background": SLATE_800, "minHeight": "100vh", "padding": "0",
            "borderRight": "1px solid #334155",
        }),

        # ── MAIN CONTENT ─────────────────────────────────────────────
        dbc.Col([
            dbc.Row(id="kpi-row", className="mb-3", style={"marginTop": "16px"}),

            dbc.Tabs([
                dbc.Tab(label="Forecast", tab_id="tab-forecast"),
                dbc.Tab(label="Heatmap", tab_id="tab-heatmap"),
                dbc.Tab(label="Metrics", tab_id="tab-metrics"),
                dbc.Tab(label="Data", tab_id="tab-data"),
            ], id="tabs", active_tab="tab-forecast", style={"marginBottom": "16px"}),

            html.Div(id="tab-content", style={
                "background": WHITE, "borderRadius": "12px",
                "border": f"1.5px solid {SLATE_100}", "padding": "16px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.04)", "minHeight": "450px",
            }),
        ], md=9, style={"background": SLATE_50, "padding": "0 24px"}),
    ]),
])


# ── Callbacks ────────────────────────────────────────────────────────

def save_uploaded(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    path = os.path.join(DATA_DIR, f"{uuid.uuid4()}_{filename}")
    with open(path, "wb") as f:
        f.write(decoded)
    return path


def load_dataframe(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df = normalize_columns(df)
    return df


@app.callback(
    Output("upload-status", "children"),
    Output("dataset-dd", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def on_upload(contents, filename):
    if not contents:
        rows = db.list_datasets()
        opts = [{"label": f"{r[1]} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r[3]))})", "value": r[0]} for r in rows]
        return no_update, opts
    p = save_uploaded(contents, filename)
    try:
        df = load_dataframe(p)
        meta = {"rows": len(df), "columns": list(df.columns)}
    except Exception as e:
        return dbc.Alert(f"Upload failed: {e}", color="danger"), no_update
    ds_id = db.add_dataset(filename, p, meta)
    rows = db.list_datasets()
    opts = [{"label": f"{r[1]} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r[3]))})", "value": r[0]} for r in rows]
    return dbc.Alert(f"Uploaded {filename}", color="success"), opts


@app.callback(Output("kpi-row", "children"), Input("dataset-dd", "value"))
def on_dataset_change(ds_id):
    if not ds_id:
        return []
    datasets = db.list_datasets()
    match = [r for r in datasets if r[0] == ds_id]
    if not match:
        return []
    path = match[0][2]
    try:
        df = load_dataframe(path)
        k = kpis(df)
    except Exception:
        return [dbc.Col(dbc.Alert("Could not load dataset.", color="warning"))]
    return [
        dbc.Col(kpi_card("Current Balance", k.get("current_balance")), md=4),
        dbc.Col(kpi_card("Avg Daily Net (30d)", k.get("avg_daily_net")), md=4),
        dbc.Col(kpi_card("Volatility (90d)", k.get("volatility")), md=4),
    ]


if long_callback_manager:
    _run_decorator = app.long_callback(
        output=Output("run-status", "children"),
        inputs=[Input("run-btn", "n_clicks")],
        state=[State("dataset-dd", "value"), State("model-dd", "value"), State("horizon", "value"),
               State("param-window", "value"), State("param-p", "value"),
               State("param-d", "value"), State("param-q", "value"), State("param-s", "value")],
        running=[(Output("run-btn", "disabled"), True, False)],
        prevent_initial_call=True,
    )
else:
    _run_decorator = app.callback(
        Output("run-status", "children"),
        Input("run-btn", "n_clicks"),
        State("dataset-dd", "value"), State("model-dd", "value"), State("horizon", "value"),
        State("param-window", "value"), State("param-p", "value"),
        State("param-d", "value"), State("param-q", "value"), State("param-s", "value"),
        prevent_initial_call=True,
    )


@_run_decorator
def run_experiment(*args):
    if long_callback_manager:
        set_progress, n, ds_id, model_name, horizon, window, p, d, q, s = args
    else:
        (n, ds_id, model_name, horizon, window, p, d, q, s) = args
        set_progress = lambda *_a, **_k: None
    if not ds_id:
        return dbc.Alert("Select a dataset first.", color="warning")
    rows = db.list_datasets()
    match = [r for r in rows if r[0] == ds_id]
    if not match:
        return dbc.Alert("Dataset not found.", color="danger")
    path = match[0][2]
    try:
        df = load_dataframe(path)
    except Exception as e:
        return dbc.Alert(f"Failed to load data: {e}", color="danger")
    params = {}
    if "Moving Average" in model_name:
        params["window"] = window or 14
    if "ARIMA" in model_name:
        params.update({"p": p or 1, "d": d or 1, "q": q or 1, "P": 0, "D": 0, "Q": 0, "s": s or 7})
    run_id = db.create_run(ds_id, model_name, params, int(horizon))
    db.update_run(run_id, status="RUNNING", started_at=time.time())
    try:
        model = registry.get(model_name)
        out = model.predict(df, int(horizon), params)
        art_dir = os.path.join(ART_DIR, run_id)
        os.makedirs(art_dir, exist_ok=True)
        out["forecast"].to_csv(os.path.join(art_dir, "forecast.csv"), index=False)
        with open(os.path.join(art_dir, "metrics.json"), "w") as f:
            json.dump(out.get("metrics", {}), f)
        db.update_run(run_id, status="DONE", finished_at=time.time(), metrics=out.get("metrics", {}))
        return dbc.Alert(f"Run {run_id[:8]} completed.", color="success")
    except Exception as e:
        db.update_run(run_id, status="FAILED", finished_at=time.time(), error=str(e))
        return dbc.Alert(f"Run failed: {e}", color="danger")


@app.callback(Output("runs-div", "children"), Input("runs-refresh", "n_intervals"))
def refresh_runs(_):
    rows = db.list_runs(limit=200)
    if not rows:
        return html.Div([
            empty_runs_table(),
            html.P("No experiments yet. Upload data and run a model.",
                    style={"color": "#64748b", "fontSize": "13px", "textAlign": "center", "marginTop": "12px"}),
        ])
    return runs_table(rows)


@app.callback(
    Output("selected-run-id", "data"),
    Input("runs-table", "selected_rows"),
    State("runs-table", "data"),
)
def select_run(selected, data):
    if not selected or not data:
        return no_update
    r = data[selected[0]]
    full = [x for x in db.list_runs(200) if x[0].startswith(r["run_id"])]
    if full:
        return full[0][0]
    return no_update


def _chart_layout(fig, height=440):
    fig.update_layout(
        margin=dict(t=10, r=10, l=10, b=10), height=height,
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Inter, -apple-system, sans-serif"),
        xaxis=dict(gridcolor=SLATE_100, showgrid=True),
        yaxis=dict(gridcolor=SLATE_100, showgrid=True, tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def render_forecast(run_id):
    if not run_id:
        return html.Div("Select a run from the history table.",
                         style={"padding": "2rem", "color": "#64748b", "textAlign": "center"})
    art_dir = os.path.join(ART_DIR, run_id)
    fc_path = os.path.join(art_dir, "forecast.csv")
    rows = db.list_runs(200)
    row = [r for r in rows if r[0] == run_id]
    if not row:
        return html.Div("Run not found.", style={"color": ROSE_600})
    row = row[0]
    ds = [d for d in db.list_datasets() if d[0] == row[1]]
    if not ds:
        return html.Div("Dataset not found.", style={"color": ROSE_600})
    try:
        df = load_dataframe(ds[0][2])
    except Exception as e:
        return html.Div(f"Failed to load data: {e}", style={"color": ROSE_600})
    fig = go.Figure()
    if "balance" in df.columns and df["balance"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["balance"], mode="lines",
                                  name="Actual Balance", line=dict(color=SLATE_800, width=2)))
    else:
        fig.add_trace(go.Scatter(x=df["date"], y=df["net_flow"].cumsum(), mode="lines",
                                  name="Cumulative", line=dict(color=SLATE_800, width=2)))
    if os.path.exists(fc_path):
        fc = pd.read_csv(fc_path, parse_dates=["date"])
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines",
                                  name="Forecast", line=dict(color=TEAL, width=2.5)))
        fig.add_traces([
            go.Scatter(x=fc["date"], y=fc["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=fc["date"], y=fc["yhat_lower"], fill="tonexty", line=dict(width=0),
                        name="95% CI", fillcolor="rgba(13,148,136,0.12)"),
        ])
    _chart_layout(fig)
    fig.update_xaxes(rangeslider_visible=True)
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def render_heatmap(run_id):
    if not run_id:
        return html.Div("Select a run to view heatmap.", style={"padding": "2rem", "color": "#64748b", "textAlign": "center"})
    row = [r for r in db.list_runs(200) if r[0] == run_id]
    if not row:
        return html.Div("Run not found.")
    ds = [d for d in db.list_datasets() if d[0] == row[0][1]]
    if not ds:
        return html.Div("Dataset not found.")
    try:
        df = load_dataframe(ds[0][2])
    except Exception as e:
        return html.Div(f"Failed to load: {e}")
    if "net_flow" not in df.columns:
        return html.Div("Heatmap needs net_flow column.", style={"color": "#64748b"})
    s = df.copy()
    s["weekday"] = s["date"].dt.weekday
    s["week"] = s["date"].dt.isocalendar().week.astype(int)
    pivot = s.pivot_table(index="week", columns="weekday", values="net_flow", aggfunc="sum").fillna(0.0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Teal",
                     labels=dict(x="Weekday (Mon=0)", y="ISO Week", color="Net flow"))
    _chart_layout(fig)
    return dcc.Graph(figure=fig, config={"displaylogo": False})


def render_metrics(run_id):
    if not run_id:
        return html.Div("Select a run to view metrics.", style={"padding": "2rem", "color": "#64748b", "textAlign": "center"})
    row = db.get_run(run_id)
    if not row:
        return html.Div("Run not found.")
    metrics = json.loads(row[8]) if row[8] else {}
    if not metrics:
        return html.Div("No metrics recorded.", style={"color": "#64748b"})
    df = pd.DataFrame([metrics])
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns], page_size=10,
        style_header={"fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                       "backgroundColor": SLATE_50, "borderBottom": f"2px solid {SLATE_100}"},
        style_cell={"fontFamily": "'Inter', sans-serif", "fontSize": "13px", "padding": "10px",
                     "fontVariantNumeric": "tabular-nums"},
    )


def render_data(run_id):
    if not run_id:
        return html.Div("Select a run to view data.", style={"padding": "2rem", "color": "#64748b", "textAlign": "center"})
    row = [r for r in db.list_runs(200) if r[0] == run_id]
    if not row:
        return html.Div("Run not found.")
    ds = [d for d in db.list_datasets() if d[0] == row[0][1]]
    if not ds:
        return html.Div("Dataset not found.")
    try:
        df = load_dataframe(ds[0][2]).copy()
    except Exception as e:
        return html.Div(f"Failed to load: {e}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns], page_size=15,
        style_table={"height": "450px", "overflowY": "auto"},
        style_header={"fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                       "backgroundColor": SLATE_50, "borderBottom": f"2px solid {SLATE_100}"},
        style_cell={"fontFamily": "'Inter', sans-serif", "fontSize": "13px", "padding": "8px 10px",
                     "fontVariantNumeric": "tabular-nums"},
    )


@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"), State("selected-run-id", "data"))
def render_tabs(active_tab, run_id):
    if active_tab == "tab-forecast":
        return render_forecast(run_id)
    if active_tab == "tab-heatmap":
        return render_heatmap(run_id)
    if active_tab == "tab-metrics":
        return render_metrics(run_id)
    if active_tab == "tab-data":
        return render_data(run_id)
    return html.Div()


if __name__ == "__main__":
    app.run_server(debug=True, port=8060)

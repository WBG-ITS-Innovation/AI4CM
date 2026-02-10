
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
# Dash long callbacks via diskcache require extra deps (dash[diskcache]) which includes psutil.
# If not installed, we gracefully fall back to synchronous callbacks.
try:
    import diskcache
    from dash.long_callback import DiskcacheLongCallbackManager

    cache = diskcache.Cache("./.cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)
except Exception:
    cache = None
    long_callback_manager = None

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="Cashflow Experiments",
    long_callback_manager=long_callback_manager,
)
server = app.server

# Ensure DB exists
db.init_db()
DATA_DIR = "data"
ART_DIR = "artifacts"
pathlib.Path(DATA_DIR).mkdir(exist_ok=True)
pathlib.Path(ART_DIR).mkdir(exist_ok=True)

def kpi_card(title, value):
    return html.Div(className="card-kpi", children=[
        html.P(title, className="kpi-title"),
        html.P(f"{value:,.0f}" if value is not None else "-", className="kpi-value")
    ])

def runs_table(rows):
    def fmt_status(s): return html.Span(s, className=f"run-status {s}")
    records = []
    for r in rows:
        run_id, dataset_id, model, params, horizon, status, started_at, finished_at, metrics, error = r
        metrics = json.loads(metrics) if metrics else {}
        records.append({
            "run_id": run_id[:8],
            "model": model,
            "horizon": horizon,
            "status": status,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
            "elapsed_s": "-" if not finished_at else f"{finished_at - started_at:.1f}",
            "mae": None if not metrics else round(metrics.get("mae", None) or 0, 2),
            "rmse": None if not metrics else round(metrics.get("rmse", None) or 0, 2),
        })
    columns=[{"name":"Run","id":"run_id"},{"name":"Model","id":"model"},{"name":"H","id":"horizon"},{"name":"Status","id":"status"},
             {"name":"Started","id":"started"},{"name":"Elapsed(s)","id":"elapsed_s"},{"name":"MAE","id":"mae"},{"name":"RMSE","id":"rmse"}]
    return dash_table.DataTable(data=records, columns=columns, page_size=8, id="runs-table",
                                row_selectable="single", style_table={"overflowY":"auto","height":"380px"})

# Layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            html.H4("Cashflow Forecast — Experiments"),
            html.P("Upload data, run models, keep history. Built for live demos."),
            dcc.Upload(id="upload-data", children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                       style={"width":"100%","height":"64px","lineHeight":"64px","borderWidth":"1px","borderStyle":"dashed",
                              "borderRadius":"6px","textAlign":"center","marginBottom":"10px"}, multiple=False),
            html.Div(id="upload-status", className="mb-2"),
            html.Label("Choose dataset"),
            dcc.Dropdown(id="dataset-dd", options=[], placeholder="Select a dataset...", className="mb-2"),
            html.Label("Model"),
            dcc.Dropdown(id="model-dd", options=[{"label":m,"value":m} for m in registry.list_models()], value="A0 • Moving Average", className="mb-2"),
            dbc.Row([
                dbc.Col([html.Label("Horizon (days)"), dcc.Slider(7, 180, 1, value=60, id="horizon")], md=12),
            ], className="mb-2"),
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([html.Label("MA Window"), dcc.Input(id="param-window", type="number", value=14)], md=6),
                        dbc.Col([html.Label("ARIMA p,d,q"), dcc.Input(id="param-p", type="number", value=1, style={"width":"30%"}),
                                                   dcc.Input(id="param-d", type="number", value=1, style={"width":"30%"}),
                                                   dcc.Input(id="param-q", type="number", value=1, style={"width":"30%"})], md=12),
                        dbc.Col([html.Label("Seasonal s"), dcc.Input(id="param-s", type="number", value=7)], md=6),
                    ])
                ], title="Model Parameters")
            ], start_collapsed=True, className="mb-2"),
            dbc.Button("Run Experiment", id="run-btn", color="primary", className="me-2"),
            html.Div(id="run-status", className="mt-2"),
            html.Hr(),
            html.H6("Experiment history"),
            html.Div(id="runs-div"),
            dcc.Interval(id="runs-refresh", interval=3000, n_intervals=0),
            dcc.Store(id="selected-run-id"),
        ], md=3, className="sidebar"),

        dbc.Col([
            dbc.Row(id="kpi-row", className="mb-3"),
            dbc.Tabs([
                dbc.Tab(label="Forecast", tab_id="tab-forecast"),
                dbc.Tab(label="Heatmap", tab_id="tab-heatmap"),
                dbc.Tab(label="Metrics", tab_id="tab-metrics"),
                dbc.Tab(label="Data", tab_id="tab-data")
            ], id="tabs", active_tab="tab-forecast"),
            html.Div(id="tab-content")
        ], md=9)
    ])
])

# helpers
def save_uploaded(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # choose extension by filename
    path = os.path.join(DATA_DIR, f"{uuid.uuid4()}_{filename}")
    with open(path, "wb") as f: f.write(decoded)
    return path

def load_dataframe(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df = normalize_columns(df)
    return df

@app.callback(Output("upload-status","children"), Output("dataset-dd","options"),
              Input("upload-data","contents"), State("upload-data","filename"))
def on_upload(contents, filename):
    if not contents: 
        # fill dataset list
        rows = db.list_datasets()
        opts = [{"label": f"{r[1]} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r[3]))})", "value": r[0]} for r in rows]
        return no_update, opts
    p = save_uploaded(contents, filename)
    # read once to get KPIs/meta
    try:
        df = load_dataframe(p)
        meta={"rows": len(df), "columns": list(df.columns)}
    except Exception as e:
        return dbc.Alert(f"Upload failed: {e}", color="danger"), no_update
    ds_id = db.add_dataset(filename, p, meta)
    rows = db.list_datasets()
    opts = [{"label": f"{r[1]} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r[3]))})", "value": r[0]} for r in rows]
    return dbc.Alert(f"Uploaded {filename} → dataset {ds_id[:8]}", color="success"), opts

@app.callback(Output("kpi-row","children"), Input("dataset-dd","value"))
def on_dataset_change(ds_id):
    if not ds_id: return []
    # find dataset path
    datasets = db.list_datasets()
    match = [r for r in datasets if r[0]==ds_id]
    if not match: return []
    path = match[0][2]
    df = load_dataframe(path)
    k = kpis(df)
    return [
        dbc.Col(kpi_card("Current Balance", k.get("current_balance")), md=3),
        dbc.Col(kpi_card("Avg Daily Net (30d)", k.get("avg_daily_net")), md=3),
        dbc.Col(kpi_card("Volatility (90d)", k.get("volatility")), md=3)
    ]

# Create and run experiment (async if available; otherwise sync)
if long_callback_manager:
    _run_decorator = app.long_callback(
        output=Output("run-status", "children"),
        inputs=[Input("run-btn", "n_clicks")],
        state=[
            State("dataset-dd", "value"),
            State("model-dd", "value"),
            State("horizon", "value"),
            State("param-window", "value"),
            State("param-p", "value"),
            State("param-d", "value"),
            State("param-q", "value"),
            State("param-s", "value"),
        ],
        running=[(Output("run-btn", "disabled"), True, False)],
        prevent_initial_call=True,
    )
else:
    _run_decorator = app.callback(
        Output("run-status", "children"),
        Input("run-btn", "n_clicks"),
        State("dataset-dd", "value"),
        State("model-dd", "value"),
        State("horizon", "value"),
        State("param-window", "value"),
        State("param-p", "value"),
        State("param-d", "value"),
        State("param-q", "value"),
        State("param-s", "value"),
        prevent_initial_call=True,
    )


@_run_decorator
def run_experiment(*args):
    # Long callback signature is (set_progress, n_clicks, ...states)
    # Normal callback signature is (n_clicks, ...states)
    if long_callback_manager:
        set_progress, n, ds_id, model_name, horizon, window, p, d, q, s = args
    else:
        (n, ds_id, model_name, horizon, window, p, d, q, s) = args
        set_progress = lambda *_a, **_k: None

    if not ds_id: 
        return dbc.Alert("Select a dataset first.", color="warning")
    rows = db.list_datasets()
    match = [r for r in rows if r[0]==ds_id]
    if not match:
        return dbc.Alert("Dataset not found.", color="danger")
    path = match[0][2]
    df = load_dataframe(path)
    # create run
    params = {}
    if "Moving Average" in model_name:
        params["window"] = window or 14
    if "ARIMA" in model_name:
        params.update({"p":p or 1, "d":d or 1, "q":q or 1, "P":0, "D":0, "Q":0, "s": s or 7})
    run_id = db.create_run(ds_id, model_name, params, int(horizon))
    db.update_run(run_id, status="RUNNING", started_at=time.time())
    set_progress(f"Run {run_id[:8]} started…")
    try:
        model = registry.get(model_name)
        out = model.predict(df, int(horizon), params)
        # save artifacts
        art_dir = os.path.join(ART_DIR, run_id)
        os.makedirs(art_dir, exist_ok=True)
        out["forecast"].to_csv(os.path.join(art_dir, "forecast.csv"), index=False)
        with open(os.path.join(art_dir, "metrics.json"), "w") as f: json.dump(out.get("metrics",{}), f)
        db.update_run(run_id, status="DONE", finished_at=time.time(), metrics=out.get("metrics",{}))
        return dbc.Alert(f"Run {run_id[:8]} completed.", color="success")
    except Exception as e:
        db.update_run(run_id, status="FAILED", finished_at=time.time(), error=str(e))
        return dbc.Alert(f"Run failed: {e}", color="danger")

@app.callback(Output("runs-div","children"), Input("runs-refresh","n_intervals"))
def refresh_runs(_):
    rows = db.list_runs(limit=200)
    return runs_table(rows)

@app.callback(Output("selected-run-id","data"), Input("runs-table","selected_rows"), State("runs-table","data"))
def select_run(selected, data):
    if not selected: return no_update
    r = data[selected[0]]
    # find full run id by prefix
    full = [x for x in db.list_runs(200) if x[0].startswith(r["run_id"])]
    if full:
        return full[0][0]
    return no_update

def render_forecast(run_id):
    if not run_id:
        return html.Div("Select a run from the history table.", style={"padding":"1rem"})
    art_dir = os.path.join(ART_DIR, run_id)
    fc_path = os.path.join(art_dir, "forecast.csv")
    rows = db.list_runs(200)
    row = [r for r in rows if r[0]==run_id][0]
    ds_id = row[1]
    # load dataset + forecast
    ds = [d for d in db.list_datasets() if d[0]==ds_id][0]
    df = load_dataframe(ds[2])
    fig = go.Figure()
    if "balance" in df.columns and df["balance"].notna().any():
        fig.add_trace(go.Scatter(x=df["date"], y=df["balance"], mode="lines", name="Balance"))
    else:
        fig.add_trace(go.Scatter(x=df["date"], y=df["net_flow"].cumsum(), mode="lines", name="Cumulative"))
    if os.path.exists(fc_path):
        fc = pd.read_csv(fc_path, parse_dates=["date"])
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines", name="Forecast"))
        fig.add_traces([
            go.Scatter(x=fc["date"], y=fc["yhat_upper"], line=dict(width=0), showlegend=False),
            go.Scatter(x=fc["date"], y=fc["yhat_lower"], fill="tonexty", line=dict(width=0), name="95% CI", opacity=0.2)
        ])
    fig.update_layout(margin=dict(t=10,r=10,l=10,b=10), height=420)
    fig.update_xaxes(rangeslider_visible=True)
    return dcc.Graph(figure=fig)

def render_heatmap(run_id):
    if not run_id:
        return html.Div("Select a run to view heatmap.")
    # load dataset of that run
    row = [r for r in db.list_runs(200) if r[0]==run_id][0]
    ds = [d for d in db.list_datasets() if d[0]==row[1]][0]
    df = load_dataframe(ds[2])
    if "net_flow" not in df.columns: return html.Div("Heatmap needs net_flow (flows or derived from balance).")
    s = df.copy()
    s["weekday"] = s["date"].dt.weekday
    s["week"] = s["date"].dt.isocalendar().week.astype(int)
    pivot = s.pivot_table(index="week", columns="weekday", values="net_flow", aggfunc="sum").fillna(0.0)
    fig = px.imshow(pivot, aspect="auto", labels=dict(x="Weekday (Mon=0)", y="ISO Week", color="Net flow"))
    fig.update_layout(margin=dict(t=10,r=10,l=10,b=10), height=420)
    return dcc.Graph(figure=fig)

def render_metrics(run_id):
    if not run_id:
        return html.Div("Select a run to view metrics.")
    row = db.get_run(run_id)
    metrics = json.loads(row[8]) if row[8] else {}
    if not metrics: return html.Div("No metrics recorded for this run.")
    df = pd.DataFrame([metrics])
    return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name":c,"id":c} for c in df.columns], page_size=10)

def render_data(run_id):
    if not run_id: return html.Div("Select a run to view data.")
    row = [r for r in db.list_runs(200) if r[0]==run_id][0]
    ds = [d for d in db.list_datasets() if d[0]==row[1]][0]
    df = load_dataframe(ds[2]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name":c,"id":c} for c in df.columns], page_size=12, style_table={'height':'420px','overflowY':'auto'})

@app.callback(Output("tab-content","children"), Input("tabs","active_tab"), State("selected-run-id","data"))
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

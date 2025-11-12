# pages/02_History.py — Enhanced History & Run Registry
from __future__ import annotations
from pathlib import Path
import re, time, json
from typing import List
import pandas as pd
import streamlit as st

try:
    from utils_frontend import list_runs, zip_outputs, collect_output_files
except Exception:
    def list_runs():
        runs_dir = Path(__file__).resolve().parents[1] / "runs"
        if not runs_dir.exists():
            return []
        return sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    def zip_outputs(out_dir: Path) -> bytes:
        import io, zipfile
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(out_dir)))
        bio.seek(0)
        return bio.read()
    def collect_output_files(out_dir: Path):
        return sorted([p for p in out_dir.rglob("*") if p.is_file()])

APPROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = APPROOT / "runs"

st.set_page_config(page_title="🕒 History", layout="wide")
st.title("🕒 Run History & Registry")

def _ago(ts: float) -> str:
    d = time.time() - ts
    if d < 60: return f"{int(d)} s ago"
    if d < 3600: return f"{int(d//60)} min ago"
    if d < 86400: return f"{int(d//3600)} h ago"
    return f"{int(d//86400)} days ago"

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

def _parse_log_times(log_path: Path):
    if not log_path.exists(): return None, None, None
    started = finished = None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in lines:
        m = TS_RE.match(ln); 
        if m: started = m.group(1); break
    for ln in reversed(lines):
        m = TS_RE.match(ln)
        if m: finished = m.group(1); break
    from datetime import datetime
    p = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") if x else None
    s = p(started); e = p(finished)
    dur = (e - s).total_seconds() if (s and e) else None
    return started, finished, dur

def _find_first(out_dir: Path, name: str):
    for base in [out_dir, out_dir/"daily", out_dir/"weekly", out_dir/"monthly"]:
        p = base / name
        if p.exists(): return p
    return None

@st.cache_data(show_spinner=False, ttl=8)
def _scan_runs() -> pd.DataFrame:
    rows = []
    for r in list_runs():
        out = r / "outputs"
        if not out.exists(): continue
        log = r / "backend_run.log"
        started, finished, dur = _parse_log_times(log)
        preds_p = _find_first(out, "predictions_long.csv")
        mets_p  = _find_first(out, "metrics_long.csv")
        lb_p    = _find_first(out, "leaderboard.csv")

        cfg = {}
        cfg_path = _find_first(out, "artifacts/config.json") or _find_first(out, "preprocess_report.json")
        if cfg_path:
            try: cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception: pass

        best_model = ""; best_mae = ""
        if lb_p:
            try:
                lb = pd.read_csv(lb_p)
                if {"model","MAE"}.issubset(lb.columns):
                    row = lb.sort_values("MAE").iloc[0]
                    best_model, best_mae = str(row["model"]), f"{float(row['MAE']):.3f}"
            except Exception:
                pass

        rows.append({
            "Run ID": r.name,
            "Finished": finished or "",
            "When": _ago(r.stat().st_mtime),
            "Family": cfg.get("family") or cfg.get("TG_FAMILY",""),
            "Variant": cfg.get("variant",""),
            "Target": cfg.get("target",""),
            "Cadence": cfg.get("cadence",""),
            "Horizon(s)": cfg.get("horizon",""),
            "Quick": str(cfg.get("quick_mode", False)).lower(),
            "Model filter": cfg.get("model_filter",""),
            "Best model (MAE)": f"{best_model} ({best_mae})" if best_model else "",
            "Outputs path": str(out),
            "Has preds": "✅" if preds_p else "—",
            "Has metrics": "✅" if mets_p else "—",
            "Has leaderboard": "✅" if lb_p else "—",
            "Duration (s)": int(dur) if dur else "",
        })
    return pd.DataFrame(rows)

# Controls
c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1: st.button("🔄 Refresh", on_click=lambda: st.cache_data.clear(), use_container_width=True)
with c2: fam = st.text_input("Filter text", value="")
with c3: only_ok = st.checkbox("Only runs with predictions", value=False)
with c4:
    visible = st.multiselect("Columns to display",
                             options=["Run ID","Finished","When","Family","Variant","Target","Cadence","Horizon(s)",
                                      "Quick","Model filter","Best model (MAE)","Outputs path","Has preds",
                                      "Has metrics","Has leaderboard","Duration (s)"],
                             default=["Run ID","Finished","Family","Target","Cadence","Horizon(s)","Best model (MAE)","Outputs path","Duration (s)"])

df = _scan_runs()
if fam.strip():
    f = fam.lower(); df = df[df.apply(lambda r: f in (" ".join(map(str, r.values))).lower(), axis=1)]
if only_ok:
    df = df[df["Has preds"] == "✅"]

st.subheader("Overview")
st.dataframe(df[visible], use_container_width=True, hide_index=True)
st.download_button("⬇️ Download overview (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="runs_overview.csv")

# Drill-down
st.markdown("---")
runs = [r for r in list_runs() if (r/"outputs").exists()]
if not runs:
    st.stop()

sel = st.selectbox("Open run", options=[r.name for r in runs], index=0)
run = RUNS_DIR / sel
out = run / "outputs"
log = run / "backend_run.log"

preds_p = _find_first(out, "predictions_long.csv")
mets_p  = _find_first(out, "metrics_long.csv")
lb_p    = _find_first(out, "leaderboard.csv")

st.subheader(f"Run {sel}")
cA, cB, cC = st.columns(3)
with cA:
    st.caption("Outputs folder")
    st.code(str(out), language="text")
with cB:
    if log.exists():
        st.caption("Log (tail)")
        st.code(log.read_text(encoding="utf-8")[-2000:], language="text")
with cC:
    st.caption("Quick downloads")
    if out.exists():
        st.download_button("⬇️ All artifacts (.zip)", data=zip_outputs(out), file_name=f"{sel}_artifacts.zip", use_container_width=True)
    if preds_p:
        st.download_button("Predictions CSV", data=preds_p.read_bytes(), file_name="predictions_long.csv", use_container_width=True)
    if mets_p:
        st.download_button("Metrics CSV", data=mets_p.read_bytes(), file_name="metrics_long.csv", use_container_width=True)
    if lb_p:
        st.download_button("Leaderboard CSV", data=lb_p.read_bytes(), file_name="leaderboard.csv", use_container_width=True)

st.markdown("---")
cols = st.columns(3)
if preds_p:
    with cols[0]:
        st.markdown("**predictions_long.csv (head)**"); st.dataframe(pd.read_csv(preds_p).head(12), use_container_width=True)
if mets_p:
    with cols[1]:
        st.markdown("**metrics_long.csv (head)**"); st.dataframe(pd.read_csv(mets_p).head(12), use_container_width=True)
if lb_p:
    with cols[2]:
        st.markdown("**leaderboard.csv (head)**"); st.dataframe(pd.read_csv(lb_p).head(12), use_container_width=True)

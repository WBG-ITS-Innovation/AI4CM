
import sqlite3, json, os, time
from typing import Optional, Dict, Any, List

DB_PATH = os.environ.get("CF_DB_PATH", "data/experiments.db")

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = connect(); cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS datasets(
        id TEXT PRIMARY KEY,
        name TEXT,
        path TEXT,
        created_at REAL,
        meta TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs(
        id TEXT PRIMARY KEY,
        dataset_id TEXT,
        model TEXT,
        params TEXT,
        horizon INTEGER,
        status TEXT,
        started_at REAL,
        finished_at REAL,
        metrics TEXT,
        error TEXT,
        FOREIGN KEY(dataset_id) REFERENCES datasets(id)
    );
    """)
    con.commit(); con.close()

def add_dataset(name:str, path:str, meta:Dict[str,Any]):
    import uuid, time
    con = connect(); cur=con.cursor()
    ds_id = str(uuid.uuid4())
    cur.execute("INSERT INTO datasets VALUES (?,?,?,?,?)",
                (ds_id, name, path, time.time(), json.dumps(meta)))
    con.commit(); con.close()
    return ds_id

def list_datasets():
    con = connect(); cur=con.cursor()
    rows = cur.execute("SELECT id,name,path,created_at,meta FROM datasets ORDER BY created_at DESC").fetchall()
    con.close()
    return rows

def create_run(dataset_id:str, model:str, params:Dict[str,Any], horizon:int):
    import uuid, time
    con = connect(); cur=con.cursor()
    run_id = str(uuid.uuid4())
    cur.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, dataset_id, model, json.dumps(params), horizon, "QUEUED", time.time(), None, None, None))
    con.commit(); con.close()
    return run_id

def update_run(run_id:str, **fields):
    con = connect(); cur=con.cursor()
    keys = []
    vals = []
    for k, v in fields.items():
        keys.append(f"{k}=?")
        if isinstance(v, (dict, list)):
            vals.append(json.dumps(v))
        else:
            vals.append(v)
    vals.append(run_id)
    cur.execute(f"UPDATE runs SET {', '.join(keys)} WHERE id=?", vals)
    con.commit(); con.close()

def get_run(run_id:str):
    con = connect(); cur=con.cursor()
    row = cur.execute("SELECT id,dataset_id,model,params,horizon,status,started_at,finished_at,metrics,error FROM runs WHERE id=?", (run_id,)).fetchone()
    con.close()
    return row

def list_runs(limit:int=100):
    con = connect(); cur=con.cursor()
    rows = cur.execute("SELECT id,dataset_id,model,params,horizon,status,started_at,finished_at,metrics,error FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return rows

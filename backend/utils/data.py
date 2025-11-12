
import pandas as pd
import numpy as np

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    date_col = None
    for c in df.columns:
        if 'date' in c:
            date_col = c; break
    if date_col is None:
        raise ValueError("No date column found. Include a 'date' column.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.rename(columns={date_col:'date'})
    if 'balance' not in df.columns:
        for alt in ['closing_balance','closing','bal','amount']:
            if alt in df.columns: df = df.rename(columns={alt:'balance'}); break
    inflow_col = next((c for c in df.columns if c in ['inflow','inflows','credit','cr']), None)
    outflow_col = next((c for c in df.columns if c in ['outflow','outflows','debit','dr']), None)
    if inflow_col and outflow_col:
        df['net_flow'] = df[inflow_col].astype(float) - df[outflow_col].astype(float)
    if 'balance' in df.columns and 'net_flow' not in df.columns:
        df['net_flow'] = df['balance'].astype(float).diff()
    df = df.set_index('date').asfreq('D')
    if 'balance' in df.columns: df['balance'] = df['balance'].ffill()
    if 'net_flow' in df.columns: df['net_flow'] = df['net_flow'].fillna(0.0)
    return df.reset_index()

def kpis(df: pd.DataFrame) -> dict:
    out = {}
    if 'balance' in df.columns:
        out['current_balance'] = float(df['balance'].iloc[-1])
    if 'net_flow' in df.columns:
        out['avg_daily_net'] = float(df['net_flow'].tail(30).mean())
        out['volatility'] = float(df['net_flow'].tail(90).std())
    return out

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

# Reutilizamos contrato/issue de F2
from engine_f2 import ValidationIssue, _issue

LEAD_TIME_DEFAULT_DAYS = 120  # v1.4 (calendario)
BUFFER_ETA_DIAS_DEFAULT = 7   # v1.4
COVERAGE_DAYS = 30            # v1.4 (no configurable)


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    if d.month == 12:
        nxt = date(d.year + 1, 1, 1)
    else:
        nxt = date(d.year, d.month + 1, 1)
    return nxt - timedelta(days=1)


def _daterange(d0: date, d1: date) -> List[date]:
    days = (d1 - d0).days
    return [d0 + timedelta(days=i) for i in range(days + 1)]


def _safe_strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Devuelve la primera columna existente en df dentro de candidates, o levanta KeyError."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontró ninguna de estas columnas: {candidates}. Disponibles: {list(df.columns)}")


def _build_daily_demand_decimal(
    proj_df: pd.DataFrame,
    mtd_df: pd.DataFrame,
    fecha_corte: date,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Identificar columnas de meses en proj_df
    month_cols: List[Tuple[object, date]] = []
    for c in proj_df.columns:
        if c in ("GRUPO", "SKU"):
            continue
        try:
            d = pd.to_datetime(str(c)).date()
            month_cols.append((c, date(d.year, d.month, 1)))
        except Exception:
            continue

    if not month_cols:
        for c in proj_df.columns:
            if c in ("GRUPO", "SKU"):
                continue
            if isinstance(c, datetime):
                d = c.date()
                month_cols.append((c, date(d.year, d.month, 1)))
            elif isinstance(c, date):
                month_cols.append((c, date(c.year, c.month, 1)))

    # De-dup + sort
    month_cols = list({(str(col), d): (col, d) for col, d in month_cols}.values())
    month_cols.sort(key=lambda x: x[1])

    mes_corte = _month_start(fecha_corte)
    month_cols = [(col, d) for col, d in month_cols if d >= mes_corte]

    # MTD agregado por SKU
    mtd = mtd_df.copy()
    mtd["SKU"] = _safe_strip_series(mtd["SKU"])
    mtd["CANTIDAD"] = pd.to_numeric(mtd["CANTIDAD"], errors="coerce").fillna(0.0)
    mtd_by_sku = mtd.groupby("SKU", as_index=True)["CANTIDAD"].sum().to_dict()

    rows = []
    totals_by_sku: Dict[str, float] = {}

    for _, r in proj_df.iterrows():
        sku = str(r["SKU"]).strip()

        for col, m0 in month_cols:
            m_end = _month_end(m0)
            projected_month = float(r[col])

            if m0 == mes_corte:
                ventas_mtd = float(mtd_by_sku.get(sku, 0.0))
                demanda_mes = max(0.0, projected_month - ventas_mtd)
                d0 = fecha_corte
                d1 = m_end
            else:
                demanda_mes = projected_month
                d0 = m0
                d1 = m_end

            days = (d1 - d0).days + 1
            if days <= 0:
                continue

            demand_daily = demanda_mes / float(days)

            for d in _daterange(d0, d1):
                rows.append({"date": d, "SKU": sku, "demand": demand_daily})
                totals_by_sku[sku] = totals_by_sku.get(sku, 0.0) + demand_daily

    dd = pd.DataFrame(rows)
    if dd.empty:
        return dd, totals_by_sku

    dd["date"] = pd.to_datetime(dd["date"])
    dd["SKU"] = _safe_strip_series(dd["SKU"])
    dd["demand"] = dd["demand"].astype(float)
    return dd, totals_by_sku


def _build_inbound_schedule(
    transit_df: pd.DataFrame,
    fecha_corte: date,
    horizon_end: date,
    buffer_eta_days: int = BUFFER_ETA_DIAS_DEFAULT,
) -> pd.DataFrame:
    if transit_df is None or len(transit_df) == 0:
        return pd.DataFrame(columns=["date", "SKU", "qty"])

    t = transit_df.copy()
    t["SKU"] = _safe_strip_series(t["SKU"])
    t["Cantidad"] = pd.to_numeric(t["Cantidad"], errors="coerce").fillna(0.0)
    t["ETA"] = pd.to_datetime(t["ETA"], errors="coerce")

    t = t.dropna(subset=["ETA"])
    t["date"] = t["ETA"] + pd.to_timedelta(buffer_eta_days, unit="D")
    t["date"] = t["date"].dt.normalize()

    d0 = pd.to_datetime(fecha_corte)
    d1 = pd.to_datetime(horizon_end)
    t = t[(t["date"] >= d0) & (t["date"] <= d1)].copy()

    inbound = (
        t.groupby(["date", "SKU"], as_index=False)["Cantidad"]
        .sum()
        .rename(columns={"Cantidad": "qty"})
    )
    return inbound


def simulate_f3_1_baseline(
    *,
    stock_df: pd.DataFrame,
    mtd_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    proyeccion_path: str,
    modulo_central_path: str,
    importaciones_path: str,
    fecha_corte: date,
) -> Tuple[pd.DataFrame, dict, List[ValidationIssue]]:
    notices: List[ValidationIssue] = []

    # Horizonte por último mes en proyección
    month_dates = []
    for c in proj_df.columns:
        if c in ("GRUPO", "SKU"):
            continue
        try:
            d = pd.to_datetime(str(c)).date()
            month_dates.append(date(d.year, d.month, 1))
        except Exception:
            if isinstance(c, (datetime, date)):
                d = c.date() if isinstance(c, datetime) else c
                month_dates.append(date(d.year, d.month, 1))

    month_dates = sorted(set([d for d in month_dates if d >= _month_start(fecha_corte)]))
    if not month_dates:
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_NO_ACTIVE_MONTHS",
            message="F3.1: no se detectaron meses activos >= MES_CORTE para simular.",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices

    horizon_end = _month_end(month_dates[-1])

    daily_demand_df, _totals = _build_daily_demand_decimal(proj_df, mtd_df, fecha_corte)
    if daily_demand_df.empty:
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_EMPTY_DAILY_DEMAND",
            message="F3.1: demanda diaria vacía (no hay filas).",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices

    inbound_df = _build_inbound_schedule(transit_df, fecha_corte, horizon_end, BUFFER_ETA_DIAS_DEFAULT)

    proj_skus = set(_safe_strip_series(proj_df["SKU"]).tolist())

    # ✅ FIX: tolerar ambos esquemas de stock_df (STOC_* o SKU/STOCK)
    st = stock_df.copy()
    sku_col = _pick_col(st, ["STOC_SKU", "SKU"])
    qty_col = _pick_col(st, ["STOC_CANTIDAD", "STOCK", "CANTIDAD"])

    st[sku_col] = _safe_strip_series(st[sku_col])
    st[qty_col] = pd.to_numeric(st[qty_col], errors="coerce").fillna(0.0)
    stock_map = st.groupby(sku_col, as_index=True)[qty_col].sum().to_dict()

    missing_stock = sorted(proj_skus - set(stock_map.keys()))
    for sku in missing_stock:
        notices.append(_issue(
            file=modulo_central_path, sheet="STOCK-2405-1426", column=sku_col, bad_rows=[],
            code="STOCK_SKU_MISSING_ASSUME_0",
            message=f"F3.1: SKU {sku} no tiene stock informado; se asume 0.",
            type_="DATA_ERROR"
        ))

    dd = daily_demand_df[daily_demand_df["SKU"].isin(proj_skus)].copy()

    all_days = pd.date_range(pd.to_datetime(fecha_corte), pd.to_datetime(horizon_end), freq="D")
    base = pd.MultiIndex.from_product([all_days, sorted(proj_skus)], names=["date", "SKU"]).to_frame(index=False)

    dd = dd.groupby(["date", "SKU"], as_index=False)["demand"].sum()
    sim = base.merge(dd, on=["date", "SKU"], how="left")
    sim["demand"] = sim["demand"].fillna(0.0)

    if inbound_df is None or inbound_df.empty:
        sim["inbound"] = 0.0
    else:
        inbound_df2 = inbound_df.copy()
        inbound_df2["date"] = pd.to_datetime(inbound_df2["date"])
        inbound_df2["SKU"] = _safe_strip_series(inbound_df2["SKU"])
        inbound_df2["qty"] = pd.to_numeric(inbound_df2["qty"], errors="coerce").fillna(0.0)
        inbound_df2 = inbound_df2.groupby(["date", "SKU"], as_index=False)["qty"].sum()
        sim = sim.merge(inbound_df2.rename(columns={"qty": "inbound"}), on=["date", "SKU"], how="left")
        sim["inbound"] = sim["inbound"].fillna(0.0)

    out_rows = []

    for sku in sorted(proj_skus):
        sku_sim = sim[sim["SKU"] == sku].sort_values("date").copy()

        on_hand = float(stock_map.get(sku, 0.0))

        for _, r in sku_sim.iterrows():
            d = r["date"]
            demand = float(r["demand"])
            inbound = float(r["inbound"])

            on_hand_start = on_hand
            on_hand_mid = on_hand_start + inbound

            fulfilled = min(demand, on_hand_mid)
            lost = max(0.0, demand - fulfilled)

            on_hand_end = on_hand_mid - fulfilled
            if on_hand_end < 0:
                on_hand_end = 0.0  # cap en cero (v1.4)

            out_rows.append({
                "date": d.date().isoformat(),
                "SKU": sku,
                "on_hand_start": on_hand_start,
                "inbound": inbound,
                "demand": demand,
                "fulfilled": fulfilled,
                "lost_sales": lost,
                "on_hand_end": on_hand_end,
            })

            on_hand = on_hand_end

    out = pd.DataFrame(out_rows)

    total_demand = float(out["demand"].sum()) if len(out) else 0.0
    total_fulfilled = float(out["fulfilled"].sum()) if len(out) else 0.0
    total_lost = float(out["lost_sales"].sum()) if len(out) else 0.0
    fill_rate = (total_fulfilled / total_demand) if total_demand > 0 else 1.0

    kpis = {
        "F3_STAGE": "F3_1_BASELINE",
        "FECHA_CORTE_EFECTIVA": fecha_corte.isoformat(),
        "HORIZON_END": horizon_end.isoformat(),
        "BUFFER_ETA_DIAS": BUFFER_ETA_DIAS_DEFAULT,
        "LEAD_TIME_DEFAULT_DAYS": LEAD_TIME_DEFAULT_DAYS,
        "COVERAGE_DAYS": COVERAGE_DAYS,
        "TOTAL_DEMAND": total_demand,
        "TOTAL_FULFILLED": total_fulfilled,
        "TOTAL_LOST_SALES": total_lost,
        "FILL_RATE": fill_rate,
    }

    return out, kpis, notices

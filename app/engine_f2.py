from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Reutilizamos el contrato de F2 (para mantener consistencia)
# En F3.1 no vamos a tirar HardValidationError por reglas de negocio nuevas;
# Si faltan inputs canónicos, eso ya lo frena F2.
from engine_f2 import ValidationIssue, _issue


LEAD_TIME_DEFAULT_DAYS = 120  # v1.4 (calendario)
BUFFER_ETA_DIAS_DEFAULT = 7   # v1.4
COVERAGE_DAYS = 30            # v1.4 (no configurable)


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    # último día del mes
    if d.month == 12:
        nxt = date(d.year + 1, 1, 1)
    else:
        nxt = date(d.year, d.month + 1, 1)
    return nxt - timedelta(days=1)


def _daterange(d0: date, d1: date) -> List[date]:
    # inclusive
    days = (d1 - d0).days
    return [d0 + timedelta(days=i) for i in range(days + 1)]


def _safe_strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _parse_lead_time_override_table(
    proyeccion_path: str,
) -> Dict[str, int]:
    """
    v1.4: "Tabla opcional: SKU -> LEAD_TIME"
    Como la especificación no fija sheet/column names en el extracto que vimos,
    implementamos detección no-invasiva:
      - si existe una hoja llamada 'LEAD_TIME' o 'PARAMETROS' o 'PARAMS'
      - y tiene columnas SKU y LEAD_TIME (case-insensitive)
    Si no existe o no matchea, devuelve {} sin error.
    """
    candidates = ["LEAD_TIME", "PARAMETROS", "PARAMS", "PARÁMETROS"]
    override: Dict[str, int] = {}

    try:
        xls = pd.ExcelFile(proyeccion_path, engine="openpyxl")
    except Exception:
        return override

    sheet = None
    for s in candidates:
        if s in xls.sheet_names:
            sheet = s
            break
    if sheet is None:
        return override

    try:
        df = pd.read_excel(proyeccion_path, sheet_name=sheet, engine="openpyxl")
    except Exception:
        return override

    cols = {c.lower().strip(): c for c in df.columns}
    if "sku" not in cols or "lead_time" not in cols:
        return override

    sku_col = cols["sku"]
    lt_col = cols["lead_time"]

    tmp = df[[sku_col, lt_col]].copy()
    tmp[sku_col] = _safe_strip_series(tmp[sku_col])
    tmp[lt_col] = pd.to_numeric(tmp[lt_col], errors="coerce")

    tmp = tmp.dropna(subset=[sku_col, lt_col])
    tmp = tmp[tmp[lt_col] >= 0]

    for _, r in tmp.iterrows():
        override[str(r[sku_col]).strip()] = int(r[lt_col])

    return override


def _build_daily_demand_decimal(
    proj_df: pd.DataFrame,
    mtd_df: pd.DataFrame,
    fecha_corte: date,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    v1.4:
      - Demanda mes en curso ajustada:
          DEMANDA_RESTANTE = max(0, PROYECTADO_MES - VENTAS_MTD)
      - DEMANDA_DIARIA en decimal
      - Simulación diaria usa decimales
      - No se redondea nada intermedio

    Construye una tabla: [date, SKU, demand] (float).
    Retorna además un dict de totales (auditoría) por SKU.
    """
    # Identificar columnas de meses (date objects) en proj_df:
    month_cols: List[Tuple[str, date]] = []
    for c in proj_df.columns:
        if c in ("GRUPO", "SKU"):
            continue
        # en F2 ya validamos que son parseables y 1er día del mes,
        # pero acá recibimos el DF ya limpio; asumimos que c es el header original.
        # Para seguridad intentamos parsear.
        try:
            d = pd.to_datetime(str(c)).date()
        except Exception:
            continue
        month_cols.append((c, date(d.year, d.month, 1)))

    # fallback si los headers vienen como datetime directamente:
    if not month_cols:
        for c in proj_df.columns:
            if c in ("GRUPO", "SKU"):
                continue
            if isinstance(c, datetime):
                d = c.date()
                month_cols.append((c, date(d.year, d.month, 1)))
            elif isinstance(c, date):
                month_cols.append((c, date(c.year, c.month, 1)))

    month_cols = list({(str(col), d): (col, d) for col, d in month_cols}.values())
    month_cols.sort(key=lambda x: x[1])

    mes_corte = _month_start(fecha_corte)
    # Filtrar meses >= MES_CORTE
    month_cols = [(col, d) for col, d in month_cols if d >= mes_corte]

    # MTD agregado por SKU
    mtd = mtd_df.copy()
    mtd["SKU"] = _safe_strip_series(mtd["SKU"])
    mtd["CANTIDAD"] = pd.to_numeric(mtd["CANTIDAD"], errors="coerce").fillna(0.0)
    mtd_by_sku = mtd.groupby("SKU", as_index=True)["CANTIDAD"].sum().to_dict()

    rows = []
    totals_by_sku: Dict[str, float] = {}

    # Para cada SKU en proyección
    for _, r in proj_df.iterrows():
        sku = str(r["SKU"]).strip()

        for col, m0 in month_cols:
            m_start = m0
            m_end = _month_end(m0)

            projected_month = float(r[col])

            # Ajuste solo para mes de corte
            if m0 == mes_corte:
                ventas_mtd = float(mtd_by_sku.get(sku, 0.0))
                demanda_mes = max(0.0, projected_month - ventas_mtd)

                # Se distribuye SOLO en los días restantes desde FECHA_CORTE (inclusive)
                d0 = fecha_corte
                d1 = m_end
            else:
                demanda_mes = projected_month
                d0 = m_start
                d1 = m_end

            days = (d1 - d0).days + 1
            if days <= 0:
                continue

            demand_daily = demanda_mes / float(days)  # decimal

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
    """
    v1.4:
      FECHA_INGRESO = ETA + BUFFER_ETA_DIAS
    Solo ESTATUS='Tránsito' ya viene filtrado desde F2.
    """
    if transit_df is None or len(transit_df) == 0:
        return pd.DataFrame(columns=["date", "SKU", "qty"])

    t = transit_df.copy()
    t["SKU"] = _safe_strip_series(t["SKU"])
    t["Cantidad"] = pd.to_numeric(t["Cantidad"], errors="coerce").fillna(0.0)
    t["ETA"] = pd.to_datetime(t["ETA"], errors="coerce")

    t = t.dropna(subset=["ETA"])
    t["date"] = t["ETA"] + pd.to_timedelta(buffer_eta_days, unit="D")
    t["date"] = t["date"].dt.normalize()

    # recortar horizonte
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
    """
    FASE 3.1:
      - Simulación diaria con decimales
      - Sin compras
      - Stock cap a 0
      - Ventas no cubiertas = perdidas
      - No arrastre de demanda
      - DEMANDA mes de corte ajustada por MTD (max(0, proj - mtd))

    Outputs:
      - simulation_df (por día y SKU)
      - kpis dict
      - notices (trazables)
    """
    notices: List[ValidationIssue] = []

    # Horizonte = último mes presente en proyección (>= MES_CORTE)
    # Buscamos el último header de mes
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
        # No debería ocurrir si F2 pasó, pero si pasa: notice técnico
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_NO_ACTIVE_MONTHS",
            message="F3.1: no se detectaron meses activos >= MES_CORTE para simular.",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices

    horizon_end = _month_end(month_dates[-1])

    # Demanda diaria decimal
    daily_demand_df, _totals = _build_daily_demand_decimal(proj_df, mtd_df, fecha_corte)
    if daily_demand_df.empty:
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_EMPTY_DAILY_DEMAND",
            message="F3.1: demanda diaria vacía (no hay filas).",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices

    # Inbound por tránsito (ETA + buffer)
    inbound_df = _build_inbound_schedule(transit_df, fecha_corte, horizon_end, BUFFER_ETA_DIAS_DEFAULT)

    # SKUs a simular = SKUs de proyección
    proj_skus = set(_safe_strip_series(proj_df["SKU"]).tolist())

    # Stock inicial por SKU (si falta, asumir 0 y notice)
    st = stock_df.copy()
    st["STOC_SKU"] = _safe_strip_series(st["STOC_SKU"])
    st["STOC_CANTIDAD"] = pd.to_numeric(st["STOC_CANTIDAD"], errors="coerce").fillna(0.0)
    stock_map = st.groupby("STOC_SKU", as_index=True)["STOC_CANTIDAD"].sum().to_dict()

    missing_stock = sorted(proj_skus - set(stock_map.keys()))
    for sku in missing_stock:
        notices.append(_issue(
            file=modulo_central_path, sheet="STOCK-2405-1426", column="STOC_SKU", bad_rows=[],
            code="STOCK_SKU_MISSING_ASSUME_0",
            message=f"F3.1: SKU {sku} no tiene stock informado; se asume 0.",
            type_="DATA_ERROR"
        ))

    # Armamos grillas por día x SKU solo para los SKUs en proyección
    # demand ya es (date, SKU)
    dd = daily_demand_df[daily_demand_df["SKU"].isin(proj_skus)].copy()

    # Asegurar rango completo de fechas para todos los SKUs (si algún SKU no tiene demanda en algún día, demanda=0)
    all_days = pd.date_range(pd.to_datetime(fecha_corte), pd.to_datetime(horizon_end), freq="D")
    base = pd.MultiIndex.from_product([all_days, sorted(proj_skus)], names=["date", "SKU"]).to_frame(index=False)

    dd = dd.groupby(["date", "SKU"], as_index=False)["demand"].sum()
    sim = base.merge(dd, on=["date", "SKU"], how="left")
    sim["demand"] = sim["demand"].fillna(0.0)

    # Inbound merge
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

    # Simulación secuencial por SKU
    out_rows = []

    for sku in sorted(proj_skus):
        sku_sim = sim[sim["SKU"] == sku].sort_values("date").copy()

        on_hand = float(stock_map.get(sku, 0.0))
        total_demand = 0.0
        total_fulfilled = 0.0
        total_lost = 0.0
        stockout_days = 0

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

            if on_hand_mid <= 0 and demand > 0:
                stockout_days += 1

            total_demand += demand
            total_fulfilled += fulfilled
            total_lost += lost

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

        # KPI por SKU (se calcula luego global)
        # (si querés top20, lo armamos en streamlit_app.py)

    out = pd.DataFrame(out_rows)

    # KPIs globales
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

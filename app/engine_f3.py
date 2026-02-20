from __future__ import annotations

from datetime import date, datetime, timedelta
from math import ceil
from typing import Dict, List, Tuple
import re

import pandas as pd

from engine_f2 import ValidationIssue, _issue

# v1.4 defaults (ahora pasan a ser defaults de UI, pero se permiten overrides por RUN)
LEAD_TIME_DEFAULT_DAYS = 120
COVERAGE_DEFAULT_DAYS = 30

# v1.4 fijo (no solicitado parametrizar)
BUFFER_ETA_DIAS_DEFAULT = 7


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
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontró ninguna de estas columnas: {candidates}. Disponibles: {list(df.columns)}")


def _build_daily_demand_decimal(
    proj_df: pd.DataFrame,
    mtd_df: pd.DataFrame,
    fecha_corte: date,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    v1.4:
      - Mes corte: demanda restante = max(0, PROY - MTD)
      - Distribución diaria en decimal (sin redondeo intermedio)
      - No arrastre de demanda
    """
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

    month_cols = list({(str(col), d): (col, d) for col, d in month_cols}.values())
    month_cols.sort(key=lambda x: x[1])

    mes_corte = _month_start(fecha_corte)
    month_cols = [(col, d) for col, d in month_cols if d >= mes_corte]

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



# ---------------------------------------------------------------------------
# Inbound multi-ESTATUS (F3.x) + auditoría diaria
# ---------------------------------------------------------------------------

_VALID_INBOUND_STATUS_ORDER = [
    "Tránsito",
    "Con reserva",
    "Lista",
    "En producción",
    "Falta depósito",
]

_EXCLUDED_STATUS = {"Entregado"}

_STATUS_TO_COL = {
    "Tránsito": "inbound_transito",
    "Con reserva": "inbound_con_reserva",
    "Lista": "inbound_lista",
    "En producción": "inbound_en_produccion",
    "Falta depósito": "inbound_falta_deposito",
}

def _norm_key(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

def _find_sheet_name(xls: pd.ExcelFile, wanted: str) -> str:
    wk = _norm_key(wanted)
    # exact normalized match
    for sh in xls.sheet_names:
        if _norm_key(sh) == wk:
            return sh
    # contains match
    for sh in xls.sheet_names:
        shk = _norm_key(sh)
        if wk in shk or shk in wk:
            return sh
    raise KeyError(f"No se encontró hoja '{wanted}' en {xls.sheet_names}")

def _col_any(df: pd.DataFrame, candidates: list[str]) -> str:
    if df is None or df.empty:
        raise KeyError(f"DataFrame vacío al buscar columnas {candidates}")
    # exact
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        k = str(c).strip().lower()
        if k in lower_map:
            return lower_map[k]
    # normalized
    norm_map = {_norm_key(c): c for c in df.columns}
    for c in candidates:
        nk = _norm_key(c)
        if nk in norm_map:
            return norm_map[nk]
    raise KeyError(f"No se encontró ninguna columna {candidates} en {list(df.columns)}")

def _parse_date_cell(x: object) -> date | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and not x.strip()):
        return None
    try:
        ts = pd.to_datetime(x, errors="coerce")
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return None
    try:
        return ts.date()
    except Exception:
        return None


def _parse_lt_cell(x: object) -> tuple[date | None, str]:
    """
    LT es FECHA. Status:
      OK / VACIO / INVALIDO / AMBIGUO

    Regla contractual: si no puede parsearse de forma segura, NO se usa (cae a defaults).
    Consideramos AMBIGUO cuando falta el año (ej: '20/3', '20-3') o cuando el string
    no contiene un año explícito y el parser podría estar infiriéndolo.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and not x.strip()):
        return None, "VACIO"

    s = str(x).strip()

    # Ambiguo típico: dd/mm sin año
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}", s):
        return None, "AMBIGUO"

    ts = None
    try:
        ts = pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        ts = pd.NaT

    if pd.isna(ts):
        return None, "INVALIDO"

    # Si es string y no tiene año explícito, lo marcamos ambiguo (evitamos supuestos)
    if isinstance(x, str) and not re.search(r"\b(19|20)\d{2}\b", s):
        return None, "AMBIGUO"

    try:
        return ts.date(), "OK"
    except Exception:
        return None, "INVALIDO"


def _parse_date_cell_with_hoy(x: object, hoy: date) -> date | None:
    """Parsea fechas permitiendo formatos dd/mm o dd-mm sin año.
    Regla: si falta año, se asume hoy.year; si queda en el pasado vs HOY, se asume año siguiente.
    Se usa SOLO para LT (que a veces viene sin año por formato de Excel).
    """
    d = _parse_date_cell(x)
    if d is not None:
        return d
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        m = re.match(r"^(\d{1,2})[\-/](\d{1,2})$", s)
        if m:
            day = int(m.group(1))
            month = int(m.group(2))
            try:
                cand = date(hoy.year, month, day)
            except Exception:
                return None
            if cand < hoy:
                try:
                    cand = date(hoy.year + 1, month, day)
                except Exception:
                    return None
            return cand
    return None

def _num_or_none(x: object) -> float | None:
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v

def _join_unique_iso(values: list[object]) -> str:
    vals = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, (date, datetime)):
            vals.append(v.date().isoformat() if isinstance(v, datetime) else v.isoformat())
        else:
            s = str(v).strip()
            if s:
                vals.append(s)
    uniq = sorted(set(vals))
    return ";".join(uniq)

def _read_defaults_from_info_comp(info_comp_path: str) -> tuple[float, float]:
    xls = pd.ExcelFile(info_comp_path, engine="openpyxl")
    sh_opts = _find_sheet_name(xls, "Opciones logisticas")
    df_opts = pd.read_excel(xls, sheet_name=sh_opts)
    prod_col = _col_any(df_opts, ["Production Time Default", "ProductionTimeDefault", "Production_Time_Default"])
    transit_col = _col_any(df_opts, ["Transit Time Default", "TransitTimeDefault", "Transit_Time_Default"])
    prod_default = _num_or_none(df_opts[prod_col].dropna().iloc[0] if len(df_opts[prod_col].dropna()) else None)
    transit_default = _num_or_none(df_opts[transit_col].dropna().iloc[0] if len(df_opts[transit_col].dropna()) else None)
    if prod_default is None:
        prod_default = 0.0
    if transit_default is None:
        transit_default = 0.0
    return float(prod_default), float(transit_default)

def _read_provider_maps(info_comp_path: str) -> tuple[dict[str, str], dict[str, tuple[float|None, float|None]], float, float]:
    xls = pd.ExcelFile(info_comp_path, engine="openpyxl")

    sh_sku = _find_sheet_name(xls, "SKU - Especificaciones")
    df_sku = pd.read_excel(xls, sheet_name=sh_sku)
    sku_col = _col_any(df_sku, ["SKU"])
    prov_col = _col_any(df_sku, ["Proveedor", "PROVEEDOR"])
    df_sku = df_sku[[sku_col, prov_col]].copy()
    df_sku[sku_col] = _safe_strip_series(df_sku[sku_col].astype(str))
    df_sku[prov_col] = _safe_strip_series(df_sku[prov_col].astype(str))
    sku_to_prov = {str(s).strip(): str(p).strip() for s, p in zip(df_sku[sku_col], df_sku[prov_col]) if str(s).strip()}

    sh_prov = _find_sheet_name(xls, "Proveedores")
    df_p = pd.read_excel(xls, sheet_name=sh_prov)
    p_name_col = _col_any(df_p, ["Proveedor", "PROVEEDOR"])
    prod_col = _col_any(df_p, ["Production Time", "ProductionTime", "Production_Time"])
    transit_col = _col_any(df_p, ["Transit Time", "TransitTime", "Transit_Time"])
    df_p = df_p[[p_name_col, prod_col, transit_col]].copy()
    df_p[p_name_col] = _safe_strip_series(df_p[p_name_col].astype(str))

    prov_to_times: dict[str, tuple[float|None, float|None]] = {}
    for _, r in df_p.iterrows():
        name = str(r[p_name_col]).strip()
        if not name:
            continue
        prod = _num_or_none(r[prod_col])
        transit = _num_or_none(r[transit_col])
        prov_to_times[name] = (prod, transit)

    prod_def, transit_def = _read_defaults_from_info_comp(info_comp_path)
    return sku_to_prov, prov_to_times, float(prod_def), float(transit_def)

def _build_inbound_multi_status_and_audit(
    *,
    importaciones_path: str,
    info_comp_path: str,
    fecha_corte: date,
    horizon_end: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[ValidationIssue]]:
    """
    Construye inbound diario (para simulación) con estados multi-ESTATUS y genera outputs de auditoría.

    Fuente única importaciones: Importaciones.xlsx → hoja IMPORTACIONES (ESTATUS, SKU, Cantidad, ETA, LT, ETD).
    Fuente única logística: Informacion Complementaria (2).xlsx (SKU→Proveedor, tiempos por proveedor y defaults).

    HOY es determinista y se define como fecha_corte (fecha de corte efectiva del RUN).
    """
    notices: list[ValidationIssue] = []

    # --- leer importaciones ---
    try:
        df = pd.read_excel(importaciones_path, sheet_name="IMPORTACIONES", engine="openpyxl")
    except Exception as e:
        notices.append(_issue(
            file=importaciones_path, sheet="IMPORTACIONES", column="(STRUCTURE)", bad_rows=[],
            code="IMPO_SHEET_READ_FAIL",
            message=f"F3 inbound: no se pudo leer hoja IMPORTACIONES: {type(e).__name__}: {e}",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notices

    # columnas mínimas
    required = ["ESTATUS", "SKU", "Cantidad"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            notices.append(_issue(
                file=importaciones_path, sheet="IMPORTACIONES", column=c, bad_rows=[],
                code="IMPO_COL_MISSING",
                message=f"F3 inbound: falta columna obligatoria en IMPORTACIONES: {c}",
                type_="DATA_ERROR"
            ))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notices

    df = df.copy()
    df["ESTATUS"] = df["ESTATUS"].astype(str).str.strip()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)

    # parse fechas (si existen)
    df["ETA_parsed"] = df["ETA"].apply(_parse_date_cell) if "ETA" in df.columns else None
    if "LT" in df.columns:
        _lt_tmp = df["LT"].apply(_parse_lt_cell)
        df["LT_parsed"] = _lt_tmp.apply(lambda t: t[0])
        df["LT_PARSE_STATUS"] = _lt_tmp.apply(lambda t: t[1])
    else:
        df["LT_parsed"] = None
        df["LT_PARSE_STATUS"] = "VACIO"
    df["ETD_parsed"] = df["ETD"].apply(_parse_date_cell) if "ETD" in df.columns else None

    # filtrar estados válidos + excluir entregado
    valid_set = set(_VALID_INBOUND_STATUS_ORDER)
    df = df[df["ESTATUS"].isin(valid_set.union(_EXCLUDED_STATUS))].copy()
    df = df[df["ESTATUS"] != "Entregado"].copy()

    # --- leer metadata / defaults ---
    try:
        sku_to_prov, prov_to_times, prod_default, transit_default = _read_provider_maps(info_comp_path)
    except Exception as e:
        notices.append(_issue(
            file=info_comp_path, sheet="(metadata)", column="(STRUCTURE)", bad_rows=[],
            code="INFO_COMP_READ_FAIL",
            message=f"F3 inbound: no se pudo leer metadata logística: {type(e).__name__}: {e}",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notices

    hoy = fecha_corte

    rows = []
    for idx, r in df.iterrows():
        est = str(r["ESTATUS"]).strip()
        sku = str(r["SKU"]).strip()
        qty = float(r["Cantidad"]) if not pd.isna(r["Cantidad"]) else 0.0

        prov_raw = sku_to_prov.get(sku, None)
        if prov_raw is None:
            prov = ""
            prov_match_status = "SKU_NO_ENCONTRADO"
        else:
            prov = str(prov_raw).strip()
            if not prov:
                prov_match_status = "PROVEEDOR_VACIO"
            else:
                prov_match_status = "MATCH_OK" if prov in prov_to_times else "PROVEEDOR_NO_EXISTE"

        regla_base = "PROVEEDOR_OK" if prov_match_status == "MATCH_OK" else f"DEFAULT_SIN_PROVEEDOR|{prov_match_status}"


        prod = None
        transit = None
        if prov:
            prod, transit = prov_to_times.get(prov, (None, None))
        prod_used = float(prod) if prod is not None else float(prod_default)
        transit_used = float(transit) if transit is not None else float(transit_default)
        prod_fallback = (prod is None)
        transit_fallback = (transit is None)

        eta_orig = r["ETA"] if "ETA" in df.columns else None
        lt_orig = r["LT"] if "LT" in df.columns else None
        eta_fecha = r.get("ETA_parsed", None)
        lt_fecha = r.get("LT_parsed", None)
        lt_parse_status = r.get("LT_PARSE_STATUS", "VACIO")

        incluido = True
        motivo_excl = ""
        regla_usada = ""
        eta_eff: date | None = None

        if est == "Tránsito":
            if eta_fecha:
                eta_eff = eta_fecha
                regla_usada = "TRANSITO_CON_ETA"
            else:
                incluido = False
                motivo_excl = "TRÁNSITO_SIN_ETA"
                regla_usada = "TRANSITO_SIN_ETA_EXCLUIDO"

        elif est in ("Con reserva", "Lista"):
            if eta_fecha:
                eta_eff = eta_fecha
                regla_usada = f"{_norm_key(est).upper()}_CON_ETA"
            else:
                eta_eff = hoy + timedelta(days=int(round(transit_used)))
                regla_usada = f"{_norm_key(est).upper()}_SIN_ETA_HOY_MAS_TRANSIT"

        elif est == "En producción":
            if eta_fecha:
                eta_eff = eta_fecha
                regla_usada = "EN_PRODUCCION_CON_ETA"
            elif lt_fecha and lt_parse_status == "OK":
                eta_eff = lt_fecha + timedelta(days=int(round(transit_used)))
                regla_usada = "EN_PRODUCCION_SIN_ETA_USA_LT_MAS_TRANSIT"
            else:
                eta_eff = hoy + timedelta(days=int(round(prod_used + transit_used)))
                regla_usada = f"EN_PRODUCCION_SIN_ETA_NI_LT_HOY_MAS_PROD_MAS_TRANSIT|LT_{lt_parse_status}"

        elif est == "Falta depósito":
            if eta_fecha:
                eta_eff = eta_fecha
                regla_usada = "FALTA_DEPOSITO_CON_ETA"
            else:
                eta_eff = hoy + timedelta(days=int(round(prod_used + transit_used)))
                regla_usada = "FALTA_DEPOSITO_SIN_ETA_HOY_MAS_PROD_MAS_TRANSIT"

        else:
            incluido = False
            motivo_excl = "ESTATUS_NO_VALIDO"
            regla_usada = "EXCLUIDO"

        # ventana horizonte: solo aplica para incluidos con ETA efectiva
        if incluido and eta_eff:
            if eta_eff < fecha_corte or eta_eff > horizon_end:
                incluido = False
                motivo_excl = "FUERA_HORIZONTE"
                regla_usada = f"{regla_usada}|FUERA_HORIZONTE"

        rows.append({
            "date": eta_eff.isoformat() if (incluido and eta_eff) else "",
            "SKU": sku,
            "Proveedor": prov,
            "PROVEEDOR_RESUELTO": prov,
            "PROVEEDOR_MATCH_STATUS": prov_match_status,
            "PROD_DIAS_USADOS": prod_used,
            "TRANSIT_DIAS_USADOS": transit_used,
            "ESTATUS": est,
            "Cantidad_inbound": qty,
            "ETA_original": eta_orig,
            "LT_original": lt_orig,
            "LT_PARSE_STATUS": lt_parse_status,
            "ETA_EFECTIVA": eta_eff.isoformat() if eta_eff else "",
            "PROD_DIAS_usado": prod_used,
            "TRANSIT_DIAS_usado": transit_used,
            "REGLA_USADA": (regla_base + "|" + regla_usada + ("|PROD_DEFAULT" if prod_fallback else "") + ("|TRANSIT_DEFAULT" if transit_fallback else "")),
            "INCLUIDO": bool(incluido),
            "MOTIVO_EXCLUSION": motivo_excl,
        })

    detail_raw = pd.DataFrame(rows)

    # Consolidar a 1 fila por (date, SKU, ESTATUS) — respetando el contrato
    def _agg_join(s: pd.Series) -> str:
        return "|".join(sorted({str(x).strip() for x in s.dropna().tolist() if str(x).strip()}))

    if len(detail_raw) > 0:
        # incluidos (date no vacío)
        inc = detail_raw[detail_raw["INCLUIDO"] == True].copy()
        exc = detail_raw[detail_raw["INCLUIDO"] == False].copy()

        agg_cols = {
            "Cantidad_inbound": "sum",
            "Proveedor": _agg_join,
            "PROVEEDOR_RESUELTO": _agg_join,
            "PROVEEDOR_MATCH_STATUS": _agg_join,
            "PROD_DIAS_USADOS": lambda s: float(pd.to_numeric(s, errors="coerce").dropna().iloc[0]) if len(pd.to_numeric(s, errors="coerce").dropna()) else float(prod_default),
            "TRANSIT_DIAS_USADOS": lambda s: float(pd.to_numeric(s, errors="coerce").dropna().iloc[0]) if len(pd.to_numeric(s, errors="coerce").dropna()) else float(transit_default),
            "ETA_original": lambda s: _join_unique_iso(s.tolist()),
            "LT_original": lambda s: _join_unique_iso(s.tolist()),
            "ETA_EFECTIVA": _agg_join,
            "PROD_DIAS_usado": lambda s: float(pd.to_numeric(s, errors="coerce").dropna().iloc[0]) if len(pd.to_numeric(s, errors="coerce").dropna()) else float(prod_default),
            "TRANSIT_DIAS_usado": lambda s: float(pd.to_numeric(s, errors="coerce").dropna().iloc[0]) if len(pd.to_numeric(s, errors="coerce").dropna()) else float(transit_default),
            "REGLA_USADA": _agg_join,
            "INCLUIDO": "max",
            "MOTIVO_EXCLUSION": _agg_join,
        }

        detail_inc = inc.groupby(["date", "SKU", "ESTATUS"], as_index=False).agg(agg_cols) if len(inc) else pd.DataFrame(columns=detail_raw.columns)
        # excluidos: consolidar por (SKU, ESTATUS) con date vacío (contrato exige columna date igualmente)
        if len(exc):
            exc2 = exc.copy()
            exc2["date"] = ""
            detail_exc = exc2.groupby(["date", "SKU", "ESTATUS"], as_index=False).agg(agg_cols)
        else:
            detail_exc = pd.DataFrame(columns=detail_inc.columns)

        detail = pd.concat([detail_inc, detail_exc], ignore_index=True)
    else:
        detail = detail_raw.copy()

    # inbound schedule para simulación (solo incluidos)
    if len(detail) and "INCLUIDO" in detail.columns:
        inc2 = detail[detail["INCLUIDO"] == True].copy()
    else:
        inc2 = pd.DataFrame()

    if len(inc2):
        inbound_schedule = inc2[["date", "SKU", "Cantidad_inbound"]].copy()
        inbound_schedule = inbound_schedule.rename(columns={"Cantidad_inbound": "qty"})
        inbound_schedule["date"] = pd.to_datetime(inbound_schedule["date"])
        inbound_schedule["SKU"] = _safe_strip_series(inbound_schedule["SKU"])
        inbound_schedule["qty"] = pd.to_numeric(inbound_schedule["qty"], errors="coerce").fillna(0.0)
        inbound_schedule = inbound_schedule.groupby(["date", "SKU"], as_index=False)["qty"].sum()
        inbound_schedule["date"] = inbound_schedule["date"].dt.date.apply(lambda d: d.isoformat())
    else:
        inbound_schedule = pd.DataFrame(columns=["date", "SKU", "qty"])

    # pivot por estado (solo incluidos) + total
    if len(inc2):
        piv = inc2.copy()
        piv["date"] = piv["date"].astype(str)
        piv["SKU"] = _safe_strip_series(piv["SKU"])
        piv["Cantidad_inbound"] = pd.to_numeric(piv["Cantidad_inbound"], errors="coerce").fillna(0.0)

        piv["col"] = piv["ESTATUS"].map(_STATUS_TO_COL).fillna("")
        piv = piv[piv["col"] != ""]
        pivot = piv.pivot_table(index=["date", "SKU"], columns="col", values="Cantidad_inbound", aggfunc="sum", fill_value=0.0).reset_index()
        for col in _STATUS_TO_COL.values():
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot["inbound_total"] = pivot[list(_STATUS_TO_COL.values())].sum(axis=1)
        pivot = pivot[["date", "SKU", "inbound_total"] + list(_STATUS_TO_COL.values())]
    else:
        pivot = pd.DataFrame(columns=["date", "SKU", "inbound_total"] + list(_STATUS_TO_COL.values()))

    # asegurar orden de columnas en detail
    detail = detail[[
        "date", "SKU", "Proveedor", "PROVEEDOR_RESUELTO", "PROVEEDOR_MATCH_STATUS", "ESTATUS", "Cantidad_inbound",
        "ETA_original", "LT_original", "ETA_EFECTIVA",
        "PROD_DIAS_usado", "TRANSIT_DIAS_usado", "PROD_DIAS_USADOS", "TRANSIT_DIAS_USADOS",
        "REGLA_USADA", "INCLUIDO", "MOTIVO_EXCLUSION"
    ]].copy()

    return inbound_schedule, detail, pivot, notices


def simulate_f3_1_baseline(
    *,
    stock_df: pd.DataFrame,
    mtd_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    proyeccion_path: str,
    modulo_central_path: str,
    importaciones_path: str,
    info_comp_path: str,
    fecha_corte: date,
    lead_time_days: int = LEAD_TIME_DEFAULT_DAYS,
    cobertura_days: int = COVERAGE_DEFAULT_DAYS,
) -> Tuple[pd.DataFrame, dict, List[ValidationIssue], dict]:
    """
    F3.1: simulación diaria baseline (sin compras)
      - decimales
      - cap stock en 0
      - ventas perdidas
      - sin arrastre de demanda

    Nota: lead_time_days y cobertura_days NO cambian F3.1 (no hay compras),
    pero se loguean en KPIs para trazabilidad del escenario.
    """
    notices: List[ValidationIssue] = []
    extra_outputs: dict = {}

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
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    horizon_end = _month_end(month_dates[-1])

    daily_demand_df, _ = _build_daily_demand_decimal(proj_df, mtd_df, fecha_corte)
    if daily_demand_df.empty:
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_EMPTY_DAILY_DEMAND",
            message="F3.1: demanda diaria vacía (no hay filas).",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    inbound_df, inbound_audit_detail, inbound_audit_pivot, inbound_notices = _build_inbound_multi_status_and_audit(
        importaciones_path=importaciones_path,
        info_comp_path=info_comp_path,
        fecha_corte=fecha_corte,
        horizon_end=horizon_end,
    )
    notices.extend(inbound_notices)
    extra_outputs["inbound_audit_daily_detail"] = inbound_audit_detail
    extra_outputs["inbound_audit_daily_pivot"] = inbound_audit_pivot

    proj_skus = set(_safe_strip_series(proj_df["SKU"]).tolist())

    # Tolerar ambos esquemas de stock_df (STOC_* o SKU/STOCK)
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
            on_hand_mid = on_hand_start + inbound  # stock al inicio del día, post inbound

            fulfilled = min(demand, on_hand_mid)
            lost = max(0.0, demand - fulfilled)

            on_hand_end = on_hand_mid - fulfilled
            if on_hand_end < 0:
                on_hand_end = 0.0  # cap en cero

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
        "LEAD_TIME_DIAS": int(lead_time_days),
        "COBERTURA_DIAS": int(cobertura_days),
        "TOTAL_DEMAND": total_demand,
        "TOTAL_FULFILLED": total_fulfilled,
        "TOTAL_LOST_SALES": total_lost,
        "FILL_RATE": fill_rate,
    }

    return out, kpis, notices, extra_outputs


def plan_purchase_f3_2_scenario(
    *,
    simulation_df: pd.DataFrame,
    fecha_corte: date,
    proyeccion_path: str,
    lead_time_days: int,
    cobertura_days: int,
) -> Tuple[pd.DataFrame, dict, List[ValidationIssue]]:
    """
    F3.2 (v1.4 + anexo operativo):
      - Una emisión planificada en FECHA_CORTE
      - Llega en FECHA_CORTE + LEAD_TIME_DIAS (escenario)
      - Al ingreso (inicio del día post inbound existente), stock debe cubrir COBERTURA_DIAS post-llegada
      - qty = ceil(max(0, demanda_cobertura - stock_al_ingreso))

    No cambia algoritmo, solo parametriza el horizonte temporal.
    """
    notices: List[ValidationIssue] = []

    if simulation_df is None or len(simulation_df) == 0:
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_2_NO_SIMULATION",
            message="F3.2: simulation_df vacío; no se puede calcular plan de compra.",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    sim = simulation_df.copy()
    sim["date"] = pd.to_datetime(sim["date"])
    sim["SKU"] = _safe_strip_series(sim["SKU"])
    for c in ["on_hand_start", "inbound", "demand"]:
        sim[c] = pd.to_numeric(sim[c], errors="coerce").fillna(0.0)

    emission_date = pd.to_datetime(fecha_corte)
    arrival_date = emission_date + pd.to_timedelta(int(lead_time_days), unit="D")
    arrival_end = arrival_date + pd.to_timedelta(int(cobertura_days) - 1, unit="D")

    min_d = sim["date"].min()
    max_d = sim["date"].max()
    if arrival_date < min_d or arrival_end > max_d:
        notices.append(_issue(
            file=proyeccion_path, sheet="GENERAL", column="(STRUCTURE)", bad_rows=[],
            code="F3_2_WINDOW_OUT_OF_HORIZON",
            message=(
                f"F3.2: ventana de cobertura [{arrival_date.date().isoformat()}..{arrival_end.date().isoformat()}] "
                f"fuera del horizonte de simulación [{min_d.date().isoformat()}..{max_d.date().isoformat()}]."
            ),
            type_="DATA_ERROR"
        ))
        return pd.DataFrame(), {
            "status": "WINDOW_OUT_OF_HORIZON",
            "EMISSION_DATE": emission_date.date().isoformat(),
            "ARRIVAL_DATE": arrival_date.date().isoformat(),
            "COBERTURA_DIAS": int(cobertura_days),
            "LEAD_TIME_DIAS": int(lead_time_days),
        }, notices, extra_outputs

    sim_arrival = sim[sim["date"] == arrival_date].copy()
    sim_arrival["stock_al_ingreso"] = sim_arrival["on_hand_start"] + sim_arrival["inbound"]
    stock_ingreso_map = sim_arrival.set_index("SKU")["stock_al_ingreso"].to_dict()

    window = sim[(sim["date"] >= arrival_date) & (sim["date"] <= arrival_end)].copy()
    demand_cov_map = window.groupby("SKU", as_index=True)["demand"].sum().to_dict()

    skus = sorted(set(sim["SKU"].unique().tolist()))
    rows = []
    for sku in skus:
        stock_ing = float(stock_ingreso_map.get(sku, 0.0))
        dem_cov = float(demand_cov_map.get(sku, 0.0))
        faltante = max(0.0, dem_cov - stock_ing)
        qty = int(ceil(faltante))

        rows.append({
            "SKU": sku,
            "FECHA_EMISION_PLANIFICADA": emission_date.date().isoformat(),
            "FECHA_LLEGADA": arrival_date.date().isoformat(),
            "LEAD_TIME_DIAS": int(lead_time_days),
            "COBERTURA_DIAS": int(cobertura_days),
            "STOCK_AL_INGRESO": stock_ing,
            "DEMANDA_COBERTURA_POST_LLEGADA": dem_cov,
            "FALTANTE": faltante,
            "QTY_A_COMPRAR": qty,
        })

    plan = pd.DataFrame(rows)

    total_qty = int(plan["QTY_A_COMPRAR"].sum()) if len(plan) else 0
    skus_compra = int((plan["QTY_A_COMPRAR"] > 0).sum()) if len(plan) else 0

    kpis = {
        "F3_STAGE": "F3_2_PURCHASE_PLAN",
        "EMISSION_DATE": emission_date.date().isoformat(),
        "ARRIVAL_DATE": arrival_date.date().isoformat(),
        "COBERTURA_DIAS": int(cobertura_days),
        "LEAD_TIME_DIAS": int(lead_time_days),
        "TOTAL_QTY_TO_BUY": total_qty,
        "SKUS_WITH_BUY_QTY": skus_compra,
    }

    return plan, kpis, notices

# ---------------------------------------------------------------------------
# F3.3 (Extensión operativa): Purchase Plan Logístico por Contenedor (40HQ)
# ---------------------------------------------------------------------------

def _sanitize_provider_name(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return "SIN_PROVEEDOR"
    # archivo-safe
    x = re.sub(r"[^\w\-]+", "_", x, flags=re.UNICODE)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "SIN_PROVEEDOR"


def plan_purchase_logistico_contenedor_40hq(
    *,
    simulation_csv_path: str | None = None,
    simulation_daily_csv_path: str | None = None,
    info_complementaria_path: str,
    proyeccion_path: str,
) -> Tuple[pd.DataFrame, dict, List[ValidationIssue]]:
    """
    F3.x (extensión operativa, NO reemplaza F3.2):
      - Fuente única timeline: outputs/simulation_daily.csv (lost_sales por día)
      - Fuente única metadata: Informacion Complementaria.xlsx
          - SKU-ESPECIFICACIONES: SKU, PROVEEDOR, VOLUMEN_M3
          - OPCIONES_LOGISTICAS: tipo == "CONTENEDOR 40 HQ" -> CAPACIDAD_M3 (=68)
      - Agrupa por proveedor y arma contenedores completos en orden temporal
      - FECHA_LLEGADA_OBJETIVO = 15 días antes del primer quiebre cubierto por el contenedor

    Output:
      proveedor, numero_contenedor, SKU, unidades, volumen_m3,
      volumen_acumulado_contenedor, fecha_inicio_quiebre, fecha_llegada_objetivo
    """
    notices: List[ValidationIssue] = []

    # Compat: algunos callers usan simulation_csv_path y otros simulation_daily_csv_path
    if simulation_csv_path is None:
        simulation_csv_path = simulation_daily_csv_path
    if not simulation_csv_path:
        raise ValueError("Falta simulation_csv_path (ruta a outputs/simulation_daily.csv)")

    def _norm_key(x: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

    def _find_sheet_name(xls: pd.ExcelFile, wanted: str, *, fallbacks: List[str] | None = None) -> str:
        wanted_keys = [_norm_key(wanted)] + ([_norm_key(f) for f in (fallbacks or [])])
        # 1) exact normalized match
        for sh in xls.sheet_names:
            if _norm_key(sh) in wanted_keys:
                return sh
        # 2) contains match (robusto ante guiones/espacios)
        for sh in xls.sheet_names:
            shk = _norm_key(sh)
            for wk in wanted_keys:
                if wk in shk or shk in wk:
                    return sh
        raise KeyError(f"No se encontró hoja '{wanted}' (ni fallbacks) en {xls.sheet_names}")

    def _col_any(df: pd.DataFrame, candidates: List[str]) -> str:
        # match exact, case-insensitive o por clave normalizada
        if df is None or df.empty:
            raise KeyError(f"DataFrame vacío al buscar columnas {candidates}")
        # exact
        for c in candidates:
            if c in df.columns:
                return c
        # case-insensitive
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for c in candidates:
            k = str(c).strip().lower()
            if k in lower_map:
                return lower_map[k]
        # normalized
        norm_map = {_norm_key(c): c for c in df.columns}
        for c in candidates:
            nk = _norm_key(c)
            if nk in norm_map:
                return norm_map[nk]
        raise KeyError(f"No se encontró ninguna columna {candidates} en {list(df.columns)}")

    def _col(df: pd.DataFrame, wanted: str) -> str:
        return _col_any(df, [wanted])

    # --- Leer simulation_daily.csv ---
    try:
        sim = pd.read_csv(simulation_csv_path)
    except Exception as e:
        notices.append(_issue(
            file=simulation_csv_path, sheet="(csv)", column="(STRUCTURE)", bad_rows=[],
            code="F3_3_SIMULATION_CSV_READ_ERROR",
            message=f"F3.3: no se pudo leer simulation_daily.csv: {type(e).__name__}: {e}",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    required_sim_cols = ["date", "SKU", "lost_sales"]
    missing = [c for c in required_sim_cols if c not in sim.columns]
    if missing:
        notices.append(_issue(
            file=simulation_csv_path, sheet="(csv)", column="(STRUCTURE)", bad_rows=[],
            code="F3_3_SIMULATION_CSV_MISSING_COLS",
            message=f"F3.3: faltan columnas en simulation_daily.csv: {missing}",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    sim["date"] = pd.to_datetime(sim["date"], errors="coerce")
    sim["SKU"] = _safe_strip_series(sim["SKU"])
    sim["lost_sales"] = pd.to_numeric(sim["lost_sales"], errors="coerce").fillna(0.0)
    sim = sim.dropna(subset=["date"]).copy()
    sim["date"] = sim["date"].dt.date

    # quedarnos solo con días con quiebre real
    sim = sim[sim["lost_sales"] > 0].copy()
    if sim.empty:
        return pd.DataFrame(), {"status": "NO_BREAKS"}, notices, extra_outputs

    # --- Leer Informacion Complementaria.xlsx ---
    try:
        xls = pd.ExcelFile(info_complementaria_path)
        meta_sheet = _find_sheet_name(
            xls,
            "SKU-ESPECIFICACIONES",
            fallbacks=["SKU - Especificaciones", "SKU-Especificaciones", "SKU Especificaciones"],
        )
        sku_meta = pd.read_excel(xls, sheet_name=meta_sheet)
    except Exception as e:
        notices.append(_issue(
            file=info_complementaria_path, sheet="SKU-ESPECIFICACIONES", column="(STRUCTURE)", bad_rows=[],
            code="F3_3_META_READ_ERROR",
            message=f"F3.3: no se pudo leer hoja SKU-ESPECIFICACIONES: {type(e).__name__}: {e}",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs


    try:
        c_sku = _col_any(sku_meta, ["SKU"])
        c_prov = _col_any(sku_meta, ["PROVEEDOR", "Proveedor"])
        c_vol = _col_any(sku_meta, ["VOLUMEN_M3", "Volumen (m3)", "Volumen m3", "Volumen_m3", "VOLUMEN"])
        c_fob = _col_any(sku_meta, ["FOB"])
    except Exception as e:
        notices.append(_issue(
            file=info_complementaria_path, sheet="SKU-ESPECIFICACIONES", column="(STRUCTURE)", bad_rows=[],
            code="F3_3_META_MISSING_COLS",
            message=f"F3.3: faltan columnas en SKU-ESPECIFICACIONES: {type(e).__name__}: {e}",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs
        notices.append(_issue(
            file=info_complementaria_path, sheet="SKU-ESPECIFICACIONES", column="(STRUCTURE)", bad_rows=[],
            code="F3_3_META_MISSING_COLS",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    sku_meta = sku_meta[[c_sku, c_prov, c_vol, c_fob]].copy()
    sku_meta = sku_meta.rename(columns={c_sku:"SKU", c_prov:"PROVEEDOR", c_vol:"VOLUMEN_M3", c_fob:"FOB"})
    sku_meta["SKU"] = _safe_strip_series(sku_meta["SKU"])
    sku_meta["PROVEEDOR"] = sku_meta["PROVEEDOR"].astype(str).str.strip()
    sku_meta["VOLUMEN_M3"] = pd.to_numeric(sku_meta["VOLUMEN_M3"], errors="coerce")
    sku_meta["FOB"] = pd.to_numeric(sku_meta["FOB"], errors="coerce")

    bad_vol = sku_meta[sku_meta["VOLUMEN_M3"].isna() | (sku_meta["VOLUMEN_M3"] <= 0)].copy()
    if not bad_vol.empty:
        bad_rows = bad_vol.index.tolist()[:200]
        notices.append(_issue(
            file=info_complementaria_path, sheet="SKU-ESPECIFICACIONES", column="VOLUMEN_M3", bad_rows=bad_rows,
            code="F3_3_BAD_VOLUMEN_M3",
            message="F3.3: hay SKUs con VOLUMEN_M3 vacío o <= 0; se excluirán del plan logístico.",
            type_="DATA_ERROR"
        ))
        sku_meta = sku_meta.drop(index=bad_vol.index).copy()

    if sku_meta.empty:
        return pd.DataFrame(), {"status": "NO_META"}, notices, extra_outputs

    # capacidad 40HQ
    capacity_m3 = None
    try:
        xls2 = pd.ExcelFile(info_complementaria_path)
        opts_sheet = _find_sheet_name(
            xls2,
            "OPCIONES_LOGISTICAS",
            fallbacks=["Opciones logisticas", "Opciones Logisticas", "OPCIONES LOGISTICAS"],
        )
        opts = pd.read_excel(xls2, sheet_name=opts_sheet)
        c_tipo = _col_any(opts, ["tipo", "Producto", "PRODUCTO"])
        c_cap = _col_any(opts, ["CAPACIDAD_M3", "Capacidad Maxima", "Capacidad Máxima", "Capacidad", "Capacidad maxima"])
        opts["_tipo_norm"] = opts[c_tipo].apply(_norm_key)
        target = _norm_key("CONTENEDOR 40 HQ")
        row = opts[opts["_tipo_norm"] == target].head(1)
        if len(row) == 0:
            row = opts[opts["_tipo_norm"].str.contains("contenedor") & opts["_tipo_norm"].str.contains("40hq")].head(1)
        if len(row) == 0:
            raise KeyError('No se encontró tipo="CONTENEDOR 40 HQ"')
        capacity_m3 = float(pd.to_numeric(row.iloc[0][c_cap], errors="coerce"))
    except Exception as e:
        notices.append(_issue(
            file=info_complementaria_path, sheet="OPCIONES_LOGISTICAS", column="(STRUCTURE)", bad_rows=[],
            code="F3_3_LOGISTIC_OPTION_MISSING",
            message=f'F3.3: no se pudo obtener CAPACIDAD_M3 para tipo="CONTENEDOR 40 HQ": {type(e).__name__}: {e}',
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    if capacity_m3 is None or not (capacity_m3 > 0):
        notices.append(_issue(
            file=info_complementaria_path, sheet="OPCIONES_LOGISTICAS", column="CAPACIDAD_M3", bad_rows=[],
            code="F3_3_BAD_CAPACITY",
            message="F3.3: CAPACIDAD_M3 inválida para CONTENEDOR 40 HQ.",
            type_="TECH_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_DATA"}, notices, extra_outputs, extra_outputs

    # --- Map SKU -> proveedor, volumen ---
    sku_to_provider = dict(zip(sku_meta["SKU"], sku_meta["PROVEEDOR"]))
    sku_to_vol = dict(zip(sku_meta["SKU"], sku_meta["VOLUMEN_M3"]))
    sku_to_fob = dict(zip(sku_meta["SKU"], sku_meta["FOB"]))

    sim = sim[sim["SKU"].isin(set(sku_to_provider.keys()))].copy()
    if sim.empty:
        notices.append(_issue(
            file=simulation_csv_path, sheet="(csv)", column="SKU", bad_rows=[],
            code="F3_3_NO_MATCHING_SKUS",
            message="F3.3: ningún SKU con quiebres coincide con SKU-ESPECIFICACIONES; no se genera plan logístico.",
            type_="DATA_ERROR"
        ))
        return pd.DataFrame(), {"status": "NO_MATCH"}, notices, extra_outputs

    sim["PROVEEDOR"] = sim["SKU"].map(sku_to_provider)
    sim["VOLUMEN_M3"] = sim["SKU"].map(sku_to_vol)
    sim["volumen_dia"] = sim["lost_sales"] * sim["VOLUMEN_M3"]

    # --- Construcción de contenedores por proveedor ---
    out_rows = []
    providers = sorted(sim["PROVEEDOR"].dropna().unique().tolist(), key=lambda x: str(x).strip().lower())

    for prov in providers:
        dfp = sim[sim["PROVEEDOR"] == prov].copy()
        if dfp.empty:
            continue

        dfp = dfp.sort_values(["date", "SKU"]).reset_index(drop=True)

        container_num = 1
        current_vol = 0.0
        container_start_date = None

        # acumulación por SKU dentro del contenedor actual
        units_by_sku: Dict[str, float] = {}

        def _flush_container():
            nonlocal container_num, current_vol, container_start_date, units_by_sku
            if current_vol + 1e-9 < capacity_m3:
                return False  # no hay contenedor completo
            # filas por SKU
                        # Totales del contenedor (para auditoría)
            volumen_total_contenedor = float(current_vol)
            fob_total_contenedor = 0.0
            for _sku, _units in units_by_sku.items():
                _fob = sku_to_fob.get(_sku)
                if _fob is not None and pd.notna(_fob):
                    fob_total_contenedor += float(_units) * float(_fob)

            # filas por SKU
            vol_acc = 0.0
            for sku, units in units_by_sku.items():
                vpu = float(sku_to_vol[sku])
                vol = float(units) * vpu
                vol_acc += vol

                fob_unitario = sku_to_fob.get(sku)
                fob_total_linea = None
                if fob_unitario is not None and pd.notna(fob_unitario):
                    fob_total_linea = float(units) * float(fob_unitario)

                out_rows.append({
                    "proveedor": str(prov).strip(),
                    "numero_contenedor": int(container_num),
                    "numero_orden": int(container_num),
                    "SKU": sku,
                    "unidades": float(units),
                    "volumen_m3": float(vol),
                    "volumen_acumulado_contenedor": float(vol_acc),
                    "volumen_total_contenedor": float(volumen_total_contenedor),
                    "fob_unitario": float(fob_unitario) if (fob_unitario is not None and pd.notna(fob_unitario)) else None,
                    "fob_total_linea": float(fob_total_linea) if fob_total_linea is not None else None,
                    "fob_total_contenedor": float(fob_total_contenedor),
                    "fecha_inicio_quiebre": container_start_date.isoformat() if container_start_date else None,
                    "fecha_llegada_objetivo": (container_start_date - timedelta(days=15)).isoformat() if container_start_date else None,
                })
# reset
            container_num += 1
            current_vol = 0.0
            container_start_date = None
            units_by_sku = {}
            return True

        # Iterar día por día (pero ya filtrado a lost_sales>0)
        for _, r in dfp.iterrows():
            d = r["date"]
            sku = r["SKU"]
            lost = float(r["lost_sales"])
            vpu = float(r["VOLUMEN_M3"])
            if lost <= 0 or vpu <= 0:
                continue

            # inicializar start date del contenedor al primer quiebre que entra
            if container_start_date is None:
                container_start_date = d

            # volumen a asignar de este registro (puede partirse entre contenedores)
            remaining_units = lost
            while remaining_units > 0:
                remaining_capacity = capacity_m3 - current_vol
                if remaining_capacity <= 1e-12:
                    # contenedor lleno, flush
                    _flush_container()
                    continue

                max_units_fit = remaining_capacity / vpu
                take_units = remaining_units if remaining_units <= max_units_fit else max_units_fit

                # acumular
                units_by_sku[sku] = units_by_sku.get(sku, 0.0) + take_units
                current_vol += take_units * vpu
                remaining_units -= take_units

                # si llegamos (o pasamos por epsilon) a capacidad, cerramos contenedor
                if current_vol + 1e-9 >= capacity_m3:
                    _flush_container()
                    # el próximo contenedor, si queda remanente, arranca en el mismo día
                    if remaining_units > 0 and container_start_date is None:
                        container_start_date = d

        # Al final, NO emitir contenedor incompleto (regla: llenar completos)
        # (se descarta remanente)

    out_df = pd.DataFrame(out_rows)

    # KPIs simples
    kpis = {
        "F3_STAGE": "F3_3_PURCHASE_PLAN_LOGISTICO_40HQ",
        "CAPACIDAD_M3": float(capacity_m3),
        "TOTAL_PROVEEDORES": int(len(out_df["proveedor"].unique())) if len(out_df) else 0,
        "TOTAL_CONTENEDORES": int(out_df.groupby(["proveedor", "numero_contenedor"]).ngroups) if len(out_df) else 0,
        "TOTAL_LINEAS": int(len(out_df)),
    }

    return out_df, kpis, notices

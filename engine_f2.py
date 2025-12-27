from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================
# Helpers (fechas)
# =========================

def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _is_month_start(d: date) -> bool:
    return d.day == 1


def _parse_excel_date_header(col) -> Optional[date]:
    if isinstance(col, datetime):
        return col.date()
    if isinstance(col, date):
        return col
    if isinstance(col, str):
        s = col.strip()
        if not s:
            return None
        try:
            dt = pd.to_datetime(s, errors="raise")
            if pd.isna(dt):
                return None
            return dt.to_pydatetime().date()
        except Exception:
            return None
    return None


# =========================
# Trazabilidad de errores
# =========================

@dataclass
class ValidationIssue:
    file: str
    sheet: str
    column: str | None
    bad_rows: list[int]
    bad_count: int
    code: str
    message: str
    type: str  # "DATA_ERROR" | "TECH_ERROR"


class HardValidationError(Exception):
    def __init__(self, issues: List[ValidationIssue]):
        super().__init__("Hard validations failed.")
        self.issues = issues


def _issue(
    *,
    file: str,
    sheet: str,
    column: str | None,
    bad_rows: list[int] | None,
    code: str,
    message: str,
    type_: str,
) -> ValidationIssue:
    rows = bad_rows or []
    return ValidationIssue(
        file=file,
        sheet=sheet,
        column=column,
        bad_rows=rows,
        bad_count=len(rows),
        code=code,
        message=message,
        type=type_,
    )


def _df_bad_rows(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    if mask is None or mask.empty:
        return []
    return df.index[mask].tolist()


# =========================
# Lectura + Validaciones (FASE 2)
# =========================

def read_stock_and_mtd(modulo_central_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[ValidationIssue]]:
    issues: List[ValidationIssue] = []

    # STOCK
    try:
        stock = pd.read_excel(modulo_central_path, sheet_name="STOCK-2405-1426", engine="openpyxl")
    except Exception as e:
        issues.append(_issue(
            file=modulo_central_path, sheet="STOCK-2405-1426", column=None, bad_rows=[],
            code="STOCK_SHEET_READ_FAIL", message=f"No se pudo leer hoja STOCK-2405-1426: {str(e)}", type_="TECH_ERROR"
        ))
        raise HardValidationError(issues)

    for col in ["STOC_SKU", "STOC_CANTIDAD"]:
        if col not in stock.columns:
            issues.append(_issue(
                file=modulo_central_path, sheet="STOCK-2405-1426", column=col, bad_rows=[],
                code="STOCK_COL_MISSING", message=f"Falta columna obligatoria en STOCK: {col}", type_="DATA_ERROR"
            ))
    if issues:
        raise HardValidationError(issues)

    stock = stock.copy()
    stock["STOC_SKU"] = stock["STOC_SKU"].astype(str).str.strip()

    qty = pd.to_numeric(stock["STOC_CANTIDAD"], errors="coerce")
    mask_neg = qty.fillna(0) < 0
    bad_rows = _df_bad_rows(stock, mask_neg)
    if bad_rows:
        issues.append(_issue(
            file=modulo_central_path, sheet="STOCK-2405-1426", column="STOC_CANTIDAD", bad_rows=bad_rows,
            code="STOCK_NEGATIVE", message="STOC_CANTIDAD tiene valores negativos (prohibido).", type_="DATA_ERROR"
        ))
        raise HardValidationError(issues)

    stock["STOC_CANTIDAD"] = qty.fillna(0)

    # MTD
    try:
        mtd = pd.read_excel(modulo_central_path, sheet_name="REPORTE_DE_PEDIDOS-3978-1426", engine="openpyxl")
    except Exception as e:
        issues.append(_issue(
            file=modulo_central_path, sheet="REPORTE_DE_PEDIDOS-3978-1426", column=None, bad_rows=[],
            code="MTD_SHEET_READ_FAIL", message=f"No se pudo leer hoja REPORTE_DE_PEDIDOS-3978-1426: {str(e)}", type_="TECH_ERROR"
        ))
        raise HardValidationError(issues)

    for col in ["FECHA_DESPACHO", "SKU", "CANTIDAD"]:
        if col not in mtd.columns:
            issues.append(_issue(
                file=modulo_central_path, sheet="REPORTE_DE_PEDIDOS-3978-1426", column=col, bad_rows=[],
                code="MTD_COL_MISSING", message=f"Falta columna obligatoria en MTD: {col}", type_="DATA_ERROR"
            ))
    if issues:
        raise HardValidationError(issues)

    mtd = mtd.copy()
    mtd["SKU"] = mtd["SKU"].astype(str).str.strip()

    parsed = pd.to_datetime(mtd["FECHA_DESPACHO"], errors="coerce")
    mask_bad_date = parsed.isna()
    bad_rows = _df_bad_rows(mtd, mask_bad_date)
    if bad_rows:
        issues.append(_issue(
            file=modulo_central_path, sheet="REPORTE_DE_PEDIDOS-3978-1426", column="FECHA_DESPACHO", bad_rows=bad_rows,
            code="MTD_DATE_PARSE", message="FECHA_DESPACHO contiene fechas no parseables.", type_="DATA_ERROR"
        ))
        raise HardValidationError(issues)

    mtd["FECHA_DESPACHO"] = parsed
    mtd["CANTIDAD"] = pd.to_numeric(mtd["CANTIDAD"], errors="coerce").fillna(0)

    return stock, mtd, []


def validate_mtd_month(mtd: pd.DataFrame, fecha_corte_efectiva: date, modulo_central_path: str) -> List[ValidationIssue]:
    mes_corte = _month_start(fecha_corte_efectiva)
    mask_bad = mtd["FECHA_DESPACHO"].dt.to_period("M") != pd.Period(mes_corte, freq="M")
    bad_rows = _df_bad_rows(mtd, mask_bad)
    if bad_rows:
        raise HardValidationError([_issue(
            file=modulo_central_path, sheet="REPORTE_DE_PEDIDOS-3978-1426", column="FECHA_DESPACHO", bad_rows=bad_rows,
            code="MTD_MONTH_MISMATCH",
            message=f"MTD contiene despachos fuera del mes de FECHA_CORTE_EFECTIVA ({mes_corte.isoformat()}).",
            type_="DATA_ERROR"
        )])
    return []


def read_and_validate_projection(proyeccion_path: str, fecha_corte_efectiva: date) -> Tuple[pd.DataFrame, Dict[Any, date], List[ValidationIssue]]:
    try:
        df = pd.read_excel(proyeccion_path, sheet_name="GENERAL", engine="openpyxl")
    except Exception as e:
        raise HardValidationError([_issue(
            file=proyeccion_path, sheet="GENERAL", column=None, bad_rows=[],
            code="PROJ_SHEET_READ_FAIL", message=f"No se pudo leer hoja GENERAL: {str(e)}", type_="TECH_ERROR"
        )])

    issues: List[ValidationIssue] = []
    for col in ["GRUPO", "SKU"]:
        if col not in df.columns:
            issues.append(_issue(
                file=proyeccion_path, sheet="GENERAL", column=col, bad_rows=[],
                code="PROJ_COL_MISSING", message=f"Falta columna base obligatoria en PROYECCION: {col}", type_="DATA_ERROR"
            ))
    if issues:
        raise HardValidationError(issues)

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["GRUPO"] = df["GRUPO"].astype(str).str.strip()

    mask_dup = df["SKU"].duplicated(keep=False)
    bad_rows = _df_bad_rows(df, mask_dup)
    if bad_rows:
        raise HardValidationError([_issue(
            file=proyeccion_path, sheet="GENERAL", column="SKU", bad_rows=bad_rows,
            code="PROJ_SKU_DUP", message="SKU duplicado en PROYECCION (prohibido).", type_="DATA_ERROR"
        )])

    month_cols = [c for c in df.columns if c not in ["GRUPO", "SKU"]]
    if not month_cols:
        raise HardValidationError([_issue(
            file=proyeccion_path, sheet="GENERAL", column=None, bad_rows=[],
            code="PROJ_NO_MONTH_COLS", message="No se detectaron columnas de meses en PROYECCION.", type_="DATA_ERROR"
        )])

    parsed_months: List[Tuple[Any, date]] = []
    for c in month_cols:
        d = _parse_excel_date_header(c)
        if d is None:
            raise HardValidationError([_issue(
                file=proyeccion_path, sheet="GENERAL", column=str(c), bad_rows=[],
                code="PROJ_MONTH_PARSE", message="Columna de mes no parseable como fecha (header inválido).", type_="DATA_ERROR"
            )])
        if not _is_month_start(d):
            raise HardValidationError([_issue(
                file=proyeccion_path, sheet="GENERAL", column=str(c), bad_rows=[],
                code="PROJ_MONTH_NOT_FIRST_DAY", message=f"Columna de mes no es primer día del mes: {d.isoformat()}", type_="DATA_ERROR"
            )])
        parsed_months.append((c, d))

    dates_only = [d for _, d in parsed_months]
    if len(set(dates_only)) != len(dates_only):
        raise HardValidationError([_issue(
            file=proyeccion_path, sheet="GENERAL", column=None, bad_rows=[],
            code="PROJ_MONTH_DUP", message="Existen meses duplicados en columnas de PROYECCION (misma fecha repetida).", type_="DATA_ERROR"
        )])

    parsed_months.sort(key=lambda x: x[1])

    mes_corte = _month_start(fecha_corte_efectiva)
    active_months = [d for _, d in parsed_months if d >= mes_corte]
    if not active_months:
        raise HardValidationError([_issue(
            file=proyeccion_path, sheet="GENERAL", column=None, bad_rows=[],
            code="PROJ_NO_ACTIVE_MONTHS", message=f"No hay meses de proyección en/desde MES_CORTE={mes_corte.isoformat()}", type_="DATA_ERROR"
        )])

    min_m, max_m = min(active_months), max(active_months)
    expected = []
    cur = min_m
    while cur <= max_m:
        expected.append(cur)
        y = cur.year + (cur.month // 12)
        m = (cur.month % 12) + 1
        cur = date(y, m, 1)

    if set(expected) != set(active_months):
        missing_months = sorted(set(expected) - set(active_months))
        raise HardValidationError([_issue(
            file=proyeccion_path, sheet="GENERAL", column=None, bad_rows=[],
            code="PROJ_MONTH_GAP",
            message=f"Falta(n) mes(es) intermedio(s) en proyección: {[d.isoformat() for d in missing_months]}",
            type_="DATA_ERROR"
        )])

    for c, _d in parsed_months:
        mask_null = df[c].isna()
        bad_rows = _df_bad_rows(df, mask_null)
        if bad_rows:
            raise HardValidationError([_issue(
                file=proyeccion_path, sheet="GENERAL", column=str(c), bad_rows=bad_rows,
                code="PROJ_DEMAND_NULL", message="Demanda NULL/vacía detectada (prohibido).", type_="DATA_ERROR"
            )])

    for c, _d in parsed_months:
        try:
            df[c] = pd.to_numeric(df[c], errors="raise")
        except Exception as e:
            raise HardValidationError([_issue(
                file=proyeccion_path, sheet="GENERAL", column=str(c), bad_rows=[],
                code="PROJ_DEMAND_NOT_NUMERIC", message=f"Demanda no numérica: {str(e)}", type_="DATA_ERROR"
            )])
        df[c] = df[c].round().astype(int)

    month_map = {col: d for col, d in parsed_months}
    return df, month_map, []


def read_and_validate_transit(importaciones_path: str) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
    try:
        df = pd.read_excel(importaciones_path, sheet_name="IMPORTACIONES", engine="openpyxl")
    except Exception as e:
        raise HardValidationError([_issue(
            file=importaciones_path, sheet="IMPORTACIONES", column=None, bad_rows=[],
            code="IMPO_SHEET_READ_FAIL", message=f"No se pudo leer hoja IMPORTACIONES: {str(e)}", type_="TECH_ERROR"
        )])

    issues: List[ValidationIssue] = []
    for col in ["ESTATUS", "SKU", "Cantidad", "ETA"]:
        if col not in df.columns:
            issues.append(_issue(
                file=importaciones_path, sheet="IMPORTACIONES", column=col, bad_rows=[],
                code="IMPO_COL_MISSING", message=f"Falta columna obligatoria en IMPORTACIONES: {col}", type_="DATA_ERROR"
            ))
    if issues:
        raise HardValidationError(issues)

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df = df.loc[df["ESTATUS"].astype(str).str.strip() == "Tránsito"].copy()

    parsed_eta = pd.to_datetime(df["ETA"], errors="coerce")
    mask_bad_eta = parsed_eta.isna()
    bad_rows = _df_bad_rows(df, mask_bad_eta)
    if bad_rows:
        raise HardValidationError([_issue(
            file=importaciones_path, sheet="IMPORTACIONES", column="ETA", bad_rows=bad_rows,
            code="IMPO_ETA_PARSE", message="ETA contiene fechas no parseables en filas Tránsito.", type_="DATA_ERROR"
        )])

    df["ETA"] = parsed_eta
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)
    return df, []

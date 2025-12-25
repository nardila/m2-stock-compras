from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# Helpers
# =========================

def _to_date_no_time(dt: datetime) -> date:
    return dt.date()


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _is_month_start(d: date) -> bool:
    return d.day == 1


def _parse_excel_date_header(col) -> Optional[date]:
    """
    Acepta headers de meses como:
      - datetime/date
      - string parseable a fecha
    Devuelve date (sin hora) o None si no parsea.
    """
    if isinstance(col, datetime):
        return col.date()
    if isinstance(col, date):
        return col
    if isinstance(col, str):
        s = col.strip()
        if not s:
            return None
        # pandas robust parse
        try:
            dt = pd.to_datetime(s, errors="raise")
            if pd.isna(dt):
                return None
            return dt.to_pydatetime().date()
        except Exception:
            return None
    return None


@dataclass
class ValidationIssue:
    level: str  # "ERROR" or "WARN" or "INFO"
    code: str
    message: str


class HardValidationError(Exception):
    def __init__(self, issues: List[ValidationIssue]):
        super().__init__("Hard validations failed.")
        self.issues = issues


# =========================
# Reading + Validations (F2)
# =========================

def read_stock_and_mtd(modulo_central_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[ValidationIssue]]:
    """
    Lee Modulo Central.xlsx:
      - STOCK-2405-1426: STOC_SKU, STOC_CANTIDAD
      - REPORTE_DE_PEDIDOS-3978-1426: FECHA_DESPACHO, SKU, CANTIDAD
    """
    issues: List[ValidationIssue] = []

    try:
        stock = pd.read_excel(
            modulo_central_path,
            sheet_name="STOCK-2405-1426",
            engine="openpyxl"
        )
    except Exception as e:
        issues.append(ValidationIssue("ERROR", "STOCK_SHEET_MISSING", f"No se pudo leer hoja STOCK-2405-1426: {e}"))
        raise HardValidationError(issues)

    required_stock_cols = {"STOC_SKU", "STOC_CANTIDAD"}
    missing = required_stock_cols - set(stock.columns)
    if missing:
        issues.append(ValidationIssue("ERROR", "STOCK_COLS_MISSING", f"Faltan columnas en STOCK: {sorted(missing)}"))
        raise HardValidationError(issues)

    # Normalización mínima (permitida): recortar espacios en SKU
    stock = stock.copy()
    stock["STOC_SKU"] = stock["STOC_SKU"].astype(str).str.strip()

    # Validación dura: cantidad >= 0
    bad_neg = stock.loc[pd.to_numeric(stock["STOC_CANTIDAD"], errors="coerce").fillna(0) < 0]
    if len(bad_neg) > 0:
        issues.append(ValidationIssue("ERROR", "STOCK_NEGATIVE", f"STOC_CANTIDAD tiene valores negativos (ej.: {len(bad_neg)} filas)."))
        raise HardValidationError(issues)

    # Casteo: cantidad numérica (sin redondear acá; F3 decide)
    stock["STOC_CANTIDAD"] = pd.to_numeric(stock["STOC_CANTIDAD"], errors="coerce").fillna(0)

    try:
        mtd = pd.read_excel(
            modulo_central_path,
            sheet_name="REPORTE_DE_PEDIDOS-3978-1426",
            engine="openpyxl"
        )
    except Exception as e:
        issues.append(ValidationIssue("ERROR", "MTD_SHEET_MISSING", f"No se pudo leer hoja REPORTE_DE_PEDIDOS-3978-1426: {e}"))
        raise HardValidationError(issues)

    required_mtd_cols = {"FECHA_DESPACHO", "SKU", "CANTIDAD"}
    missing = required_mtd_cols - set(mtd.columns)
    if missing:
        issues.append(ValidationIssue("ERROR", "MTD_COLS_MISSING", f"Faltan columnas en MTD: {sorted(missing)}"))
        raise HardValidationError(issues)

    mtd = mtd.copy()
    mtd["SKU"] = mtd["SKU"].astype(str).str.strip()
    # FECHA_DESPACHO parseable
    mtd["FECHA_DESPACHO"] = pd.to_datetime(mtd["FECHA_DESPACHO"], errors="coerce")
    if mtd["FECHA_DESPACHO"].isna().any():
        issues.append(ValidationIssue("ERROR", "MTD_DATE_PARSE", "FECHA_DESPACHO contiene fechas no parseables."))
        raise HardValidationError(issues)

    mtd["CANTIDAD"] = pd.to_numeric(mtd["CANTIDAD"], errors="coerce").fillna(0)

    return stock, mtd, issues


def validate_mtd_month(mtd: pd.DataFrame, fecha_corte_efectiva: date) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    mes_corte = _month_start(fecha_corte_efectiva)

    # Mes de FECHA_DESPACHO debe ser el mismo mes que FECHA_CORTE_EFECTIVA
    mtd_months = mtd["FECHA_DESPACHO"].dt.to_period("M").astype(str).unique().tolist()
    # Permitimos que haya filas del mismo mes únicamente
    bad = mtd.loc[mtd["FECHA_DESPACHO"].dt.to_period("M") != pd.Period(mes_corte, freq="M")]
    if len(bad) > 0:
        issues.append(ValidationIssue(
            "ERROR",
            "MTD_MONTH_MISMATCH",
            f"MTD contiene despachos fuera del mes de FECHA_CORTE_EFECTIVA ({mes_corte.isoformat()}). Filas inválidas: {len(bad)}."
        ))
        raise HardValidationError(issues)

    # INFO
    issues.append(ValidationIssue("INFO", "MTD_MONTH_OK", f"MTD verificado: mes={mes_corte.isoformat()} (valores únicos detectados: {mtd_months})."))
    return issues


def read_and_validate_projection(proyeccion_path: str, fecha_corte_efectiva: date) -> Tuple[pd.DataFrame, Dict[str, date], List[ValidationIssue]]:
    """
    Lee PROYECCION - SKU  MAYO 2026.xlsx hoja GENERAL:
    - GRUPO, SKU obligatorios
    - columnas de meses: fechas parseables y primer día del mes
    Validaciones:
    - meses anteriores a MES_CORTE -> ignorados y logueados
    - meses duplicados -> error
    - meses no contiguos entre min y max -> error
    - demanda NULL -> error
    - SKU duplicado (fila) -> error
    """
    issues: List[ValidationIssue] = []
    mes_corte = _month_start(fecha_corte_efectiva)

    try:
        df = pd.read_excel(
            proyeccion_path,
            sheet_name="GENERAL",
            engine="openpyxl"
        )
    except Exception as e:
        issues.append(ValidationIssue("ERROR", "PROJ_SHEET_MISSING", f"No se pudo leer hoja GENERAL: {e}"))
        raise HardValidationError(issues)

    required = {"GRUPO", "SKU"}
    missing = required - set(df.columns)
    if missing:
        issues.append(ValidationIssue("ERROR", "PROJ_COLS_MISSING", f"Faltan columnas base en PROYECCION: {sorted(missing)}"))
        raise HardValidationError(issues)

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["GRUPO"] = df["GRUPO"].astype(str).str.strip()

    # Duplicados SKU
    dup = df["SKU"].duplicated(keep=False)
    if dup.any():
        issues.append(ValidationIssue("ERROR", "PROJ_SKU_DUP", f"SKU duplicado en PROYECCION. Cantidad de filas duplicadas: {int(dup.sum())}."))
        raise HardValidationError(issues)

    # Detectar columnas de meses (todas excepto GRUPO, SKU)
    month_cols = [c for c in df.columns if c not in ["GRUPO", "SKU"]]
    if len(month_cols) == 0:
        issues.append(ValidationIssue("ERROR", "PROJ_NO_MONTH_COLS", "No se detectaron columnas de meses en PROYECCION."))
        raise HardValidationError(issues)

    parsed_months: List[Tuple[str, date]] = []
    for c in month_cols:
        d = _parse_excel_date_header(c)
        if d is None:
            issues.append(ValidationIssue("ERROR", "PROJ_MONTH_PARSE", f"Columna de mes no parseable como fecha: {repr(c)}"))
            raise HardValidationError(issues)
        if not _is_month_start(d):
            issues.append(ValidationIssue("ERROR", "PROJ_MONTH_NOT_FIRST_DAY", f"Columna de mes no es primer día del mes: {repr(c)} -> {d.isoformat()}"))
            raise HardValidationError(issues)
        parsed_months.append((c, d))

    # Duplicados de mes (misma fecha)
    dates_only = [d for _, d in parsed_months]
    if len(set(dates_only)) != len(dates_only):
        issues.append(ValidationIssue("ERROR", "PROJ_MONTH_DUP", "Existen meses duplicados en las columnas de PROYECCION."))
        raise HardValidationError(issues)

    # Ordenar por fecha
    parsed_months.sort(key=lambda x: x[1])

    # Meses anteriores a MES_CORTE: se ignoran (pero NO se eliminan del DF; solo se reporta para F3)
    ignored = [d for _, d in parsed_months if d < mes_corte]
    if ignored:
        issues.append(ValidationIssue("INFO", "PROJ_MONTHS_IGNORED", f"Meses ignorados por ser anteriores a MES_CORTE={mes_corte.isoformat()}: {[x.isoformat() for x in ignored]}"))

    # Validación de contigüidad SOLO sobre meses >= mes_corte
    active_months = [d for _, d in parsed_months if d >= mes_corte]
    if len(active_months) == 0:
        issues.append(ValidationIssue("ERROR", "PROJ_NO_ACTIVE_MONTHS", f"No hay meses de proyección en/desde MES_CORTE={mes_corte.isoformat()}."))
        raise HardValidationError(issues)

    min_m = min(active_months)
    max_m = max(active_months)

    # Construir secuencia esperada de meses contiguos
    expected = []
    cur = min_m
    while cur <= max_m:
        expected.append(cur)
        # sumar 1 mes
        y = cur.year + (cur.month // 12)
        m = (cur.month % 12) + 1
        cur = date(y, m, 1)

    if set(expected) != set(active_months):
        missing_months = sorted(set(expected) - set(active_months))
        issues.append(ValidationIssue("ERROR", "PROJ_MONTH_GAP", f"Falta(n) mes(es) intermedio(s) en proyección: {[d.isoformat() for d in missing_months]}"))
        raise HardValidationError(issues)

    # Validación de contenido: NULL/vacío -> error (0 válido)
    # Revisamos solo columnas de meses (todas), porque la regla es general.
    for c, _d in parsed_months:
        if df[c].isna().any():
            issues.append(ValidationIssue("ERROR", "PROJ_DEMAND_NULL", f"Demanda NULL/vacía detectada en columna {repr(c)}."))
            raise HardValidationError(issues)

    # Redondeo de decimales a entero más cercano: esto ES regla de v1.4 para demanda.
    # NO es optimización: es contrato.
    for c, _d in parsed_months:
        df[c] = pd.to_numeric(df[c], errors="raise")
        df[c] = df[c].round().astype(int)

    month_map = {col: d for col, d in parsed_months}
    issues.append(ValidationIssue("INFO", "PROJ_OK", f"Proyección OK. Meses activos: {len(active_months)} (desde {min_m.isoformat()} a {max_m.isoformat()})."))

    return df, month_map, issues


def read_and_validate_transit(importaciones_path: str) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    try:
        df = pd.read_excel(
            importaciones_path,
            sheet_name="IMPORTACIONES",
            engine="openpyxl"
        )
    except Exception as e:
        issues.append(ValidationIssue("ERROR", "IMPO_SHEET_MISSING", f"No se pudo leer hoja IMPORTACIONES: {e}"))
        raise HardValidationError(issues)

    # Requerimos columna ESTATUS para filtrar
    required = {"ESTATUS", "SKU", "Cantidad", "ETA"}
    missing = required - set(df.columns)
    if missing:
        issues.append(ValidationIssue("ERROR", "IMPO_COLS_MISSING", f"Faltan columnas en IMPORTACIONES: {sorted(missing)}"))
        raise HardValidationError(issues)

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Filtrar solo Tránsito
    df = df.loc[df["ESTATUS"].astype(str).str.strip() == "Tránsito"].copy()

    # ETA parseable (aunque luego F3 sume buffer)
    df["ETA"] = pd.to_datetime(df["ETA"], errors="coerce")
    if df["ETA"].isna().any():
        issues.append(ValidationIssue("ERROR", "IMPO_ETA_PARSE", "ETA contiene fechas no parseables en filas Tránsito."))
        raise HardValidationError(issues)

    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)

    issues.append(ValidationIssue("INFO", "IMPO_OK", f"Importaciones en tránsito leídas: {len(df)} filas."))
    return df, issues

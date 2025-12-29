import os
import io
import json
import zipfile
from datetime import datetime, date, timedelta
import hashlib
import secrets
import streamlit as st
import pandas as pd

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from engine_f2 import (
    HardValidationError,
    ValidationIssue,
    read_stock_and_mtd,
    validate_mtd_month,
    read_and_validate_projection,
    read_and_validate_transit,
)

from engine_f3 import (
    simulate_f3_1_baseline,
    plan_purchase_f3_2_scenario,
    LEAD_TIME_DEFAULT_DAYS,
    COVERAGE_DEFAULT_DAYS,
)

RUNS_DIR = "runs"

# Heatmap v1.0.1: inbound = stock inicial + tránsito (ETA + 7) según F3
BUFFER_ETA_DIAS = 7


def ensure_dirs():
    os.makedirs(RUNS_DIR, exist_ok=True)


def now_ts():
    return datetime.now()


def make_run_id(dt: datetime) -> str:
    return f"RUN_{dt.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_uploaded_files(run_path: str, uploaded_files):
    inputs_dir = os.path.join(run_path, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    saved = []
    for uf in uploaded_files:
        content = uf.getvalue()
        file_hash = sha256_bytes(content)
        dst = os.path.join(inputs_dir, uf.name)
        with open(dst, "wb") as f:
            f.write(content)
        saved.append({
            "original_name": uf.name,
            "size_bytes": len(content),
            "sha256": file_hash,
            "stored_path": dst.replace("\\", "/"),
        })
    return saved


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# CSV Output Contract (F3.x)
# - max 2 decimales
# - redondeo estándar (half-up)
# - sin notación científica
# - UTF-8
# - decimal '.'
# - NO redondear cálculos internos: solo copia exportada
# -----------------------------

def _round_half_up_2dp(x):
    if x is None:
        return x
    try:
        if pd.isna(x):
            return x
    except Exception:
        pass

    try:
        d = Decimal(str(x))
        return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, TypeError):
        return x


def _prepare_f3_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy(deep=True)
    num_cols = export_df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return export_df

    export_df[num_cols] = export_df[num_cols].applymap(_round_half_up_2dp)
    for c in num_cols:
        export_df[c] = pd.to_numeric(export_df[c], errors="coerce")
    return export_df


def write_csv_f3(path: str, df: pd.DataFrame):
    export_df = _prepare_f3_csv_export(df)
    export_df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format="%.2f",   # no sci + 2 decimales homogéneos
        lineterminator="\n"
    )


def make_zip_bytes(run_path: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(run_path):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, run_path)
                z.write(full, arcname=rel.replace("\\", "/"))
    mem.seek(0)
    return mem.read()


def issues_to_dict(issues: list[ValidationIssue]):
    return [{
        "file": i.file,
        "sheet": i.sheet,
        "column": i.column,
        "bad_rows": i.bad_rows,
        "bad_count": i.bad_count,
        "code": i.code,
        "message": i.message,
        "type": i.type,
    } for i in issues]


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def to_json_safe_month_map(month_map: dict) -> dict:
    safe = {}
    for col_header, d in month_map.items():
        safe[month_key(d)] = {"date": d.isoformat(), "source_col_header": str(col_header)}
    return safe


def build_run_log_base(run_id: str, created_at: datetime, input_files_meta, fecha_corte_override_iso: str | None,
                       lead_time_days: int, cobertura_days: int):
    fecha_corte_default = created_at.date().isoformat()
    fecha_corte_efectiva = fecha_corte_override_iso or fecha_corte_default
    return {
        "RUN_ID": run_id,
        "CREATED_AT": created_at.isoformat(timespec="seconds"),
        "FECHA_CORTE_DEFAULT": fecha_corte_default,
        "FECHA_CORTE_OVERRIDE": fecha_corte_override_iso,
        "FECHA_CORTE_EFECTIVA": fecha_corte_efectiva,
        "OVERRIDE_ACTIVO": bool(fecha_corte_override_iso),

        "PARAMETROS_RUN": {
            "FECHA_CORTE_OVERRIDE": fecha_corte_override_iso,
            "LEAD_TIME_DIAS": int(lead_time_days),
            "COBERTURA_DIAS": int(cobertura_days),
        },

        "INPUT_FILES": input_files_meta,
        "STATUS": None,
        "VALIDATIONS": [],
        "PARAMS_EFECTIVOS": {},
        "COUNTS": {},
        "NOTES": "F2: validaciones duras. F3: escenarios por RUN (LT/cobertura) + outputs CSV 2 decimales. Heatmap: diagnóstico v1.0.1.",
        "F3": {
            "STATUS": None,
            "KPIS_F3_1": {},
            "KPIS_F3_2": {},
        },
    }


def tech_issue(code: str, message: str):
    return {
        "file": "(runtime)",
        "sheet": "(runtime)",
        "column": None,
        "bad_rows": [],
        "bad_count": 0,
        "code": code,
        "message": message,
        "type": "TECH_ERROR",
    }


# -----------------------------
# Heatmap de Cobertura v1.0.1 (CANÓNICO)
# - Usa SOLO: SKUs en proyección + stock inicial + importaciones en "Tránsito"
# - Métrica: cubiertas/proyectadas por mes, cobertura % sin decimales (0..100)
# - INACTIVO si proyectadas del mes = 0 (celda celeste, sin %)
# - Encabezado mes: "Mes AAAA — cubiertas / proyectadas"
# - Orden UI: alfabético o por criticidad (menor cobertura promedio primero)
# -----------------------------

def _spanish_month_label(d: date) -> str:
    # "Mar 2026" como en el ejemplo del PDF
    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    return f"{months[d.month - 1]} {d.year}"


def _month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


def _parse_months_from_month_map(month_map: dict) -> list[date]:
    # month_map viene de F2: col_header -> date(YYYY,MM,1)
    months = sorted({date(v.year, v.month, 1) for v in month_map.values()})
    return months


def _stock_map_from_stock_df(stock_df: pd.DataFrame) -> dict[str, float]:
    st = stock_df.copy()
    # tolerar esquemas
    sku_col = "STOC_SKU" if "STOC_SKU" in st.columns else ("SKU" if "SKU" in st.columns else None)
    qty_col = "STOC_CANTIDAD" if "STOC_CANTIDAD" in st.columns else ("STOCK" if "STOCK" in st.columns else ("CANTIDAD" if "CANTIDAD" in st.columns else None))
    if sku_col is None or qty_col is None:
        return {}

    st[sku_col] = st[sku_col].astype(str).str.strip()
    st[qty_col] = pd.to_numeric(st[qty_col], errors="coerce").fillna(0.0)
    return st.groupby(sku_col, as_index=True)[qty_col].sum().to_dict()


def _inbound_by_month_from_transit(transit_df: pd.DataFrame, months: list[date]) -> pd.DataFrame:
    """
    inbound considerado (v1.0.1): importaciones estado "Tránsito"
    fecha de ingreso: ETA + 7 (buffer) como F3.
    Agregamos por (SKU, mes) según mes calendario de la fecha de ingreso.
    """
    if transit_df is None or len(transit_df) == 0:
        return pd.DataFrame(columns=["SKU", "MONTH", "INBOUND"])

    t = transit_df.copy()
    # columnas esperadas desde F2: SKU, Cantidad, ETA
    if "SKU" not in t.columns or "Cantidad" not in t.columns or "ETA" not in t.columns:
        return pd.DataFrame(columns=["SKU", "MONTH", "INBOUND"])

    t["SKU"] = t["SKU"].astype(str).str.strip()
    t["Cantidad"] = pd.to_numeric(t["Cantidad"], errors="coerce").fillna(0.0)
    t["ETA"] = pd.to_datetime(t["ETA"], errors="coerce")
    t = t.dropna(subset=["ETA"]).copy()

    t["INGRESO"] = (t["ETA"] + pd.to_timedelta(BUFFER_ETA_DIAS, unit="D")).dt.date
    t["MONTH"] = t["INGRESO"].apply(lambda d: date(d.year, d.month, 1))

    months_set = set(months)
    t = t[t["MONTH"].isin(months_set)].copy()

    g = t.groupby(["SKU", "MONTH"], as_index=False)["Cantidad"].sum()
    g = g.rename(columns={"Cantidad": "INBOUND"})
    return g


def build_heatmap_cobertura_v101(
    proj_df: pd.DataFrame,
    month_map: dict,
    stock_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    order_mode: str,
):
    """
    Devuelve:
      skus (list[str])
      months (list[date])
      coverage_pct (np.ndarray float, shape [S,K]) con NaN para INACTIVO
      state (np.ndarray int categories) 0=INACTIVO,1=ROJO,2=NARANJA,3=VERDE
      month_headers (list[str]) "Mes AAAA — cubiertas / proyectadas"
    """
    # Universo: solo SKUs en proyección
    if proj_df is None or len(proj_df) == 0 or "SKU" not in proj_df.columns:
        return [], [], np.array([[]]), np.array([[]], dtype=int), []

    months = _parse_months_from_month_map(month_map)
    if not months:
        return [], [], np.array([[]]), np.array([[]], dtype=int), []

    p = proj_df.copy()
    p["SKU"] = p["SKU"].astype(str).str.strip()
    skus = sorted(p["SKU"].unique().tolist())

    # Mapa de proyección mensual por SKU y mes
    # month_map: col_header -> date(YYYY,MM,1)
    # Armamos dict: (SKU, month) -> projected_units
    proj_by = {}
    for col, m in month_map.items():
        if col not in p.columns:
            continue
        vals = pd.to_numeric(p[col], errors="coerce").fillna(0.0)
        for sku, v in zip(p["SKU"].tolist(), vals.tolist()):
            proj_by[(sku, date(m.year, m.month, 1))] = proj_by.get((sku, date(m.year, m.month, 1)), 0.0) + float(v)

    stock_map = _stock_map_from_stock_df(stock_df)

    inbound_month_df = _inbound_by_month_from_transit(transit_df, months)
    inbound_by = {}
    if len(inbound_month_df) > 0:
        for _, r in inbound_month_df.iterrows():
            inbound_by[(r["SKU"], r["MONTH"])] = inbound_by.get((r["SKU"], r["MONTH"]), 0.0) + float(r["INBOUND"])

    S = len(skus)
    K = len(months)

    coverage = np.full((S, K), np.nan, dtype=float)
    state = np.zeros((S, K), dtype=int)  # 0=INACTIVO

    # Para encabezados: total cubiertas/proyectadas por mes
    total_proj = {m: 0.0 for m in months}
    total_cov = {m: 0.0 for m in months}

    for i, sku in enumerate(skus):
        available = float(stock_map.get(sku, 0.0))

        for j, m in enumerate(months):
            proj_units = float(proj_by.get((sku, m), 0.0))
            total_proj[m] += proj_units

            # sumar inbound del mes (ingreso en ese mes)
            available += float(inbound_by.get((sku, m), 0.0))

            if proj_units <= 0:
                # INACTIVO (celeste, sin %)
                state[i, j] = 0
                continue

            covered_units = min(available, proj_units)
            available -= covered_units
            total_cov[m] += covered_units

            pct = (covered_units / proj_units) * 100.0 if proj_units > 0 else 0.0
            if pct < 0:
                pct = 0.0
            if pct > 100:
                pct = 100.0

            coverage[i, j] = pct

            # estados visuales
            if pct >= 80:
                state[i, j] = 3  # verde
            elif pct >= 40:
                state[i, j] = 2  # naranja
            else:
                state[i, j] = 1  # rojo

    # Orden UI:
    if order_mode.startswith("Criticidad"):
        # menor cobertura promedio primero (solo meses activos)
        avg_cov = []
        for i, sku in enumerate(skus):
            vals = coverage[i, :]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                avg = 0.0
            else:
                avg = float(np.mean(vals))
            avg_cov.append((avg, sku, i))
        avg_cov.sort(key=lambda x: (x[0], x[1]))
        order_idx = [x[2] for x in avg_cov]
        skus = [skus[i] for i in order_idx]
        coverage = coverage[order_idx, :]
        state = state[order_idx, :]

    # Encabezados de mes: "Mes AAAA — cubiertas / proyectadas"
    headers = []
    for m in months:
        label = _spanish_month_label(m)
        cov_u = int(round(total_cov[m]))
        proj_u = int(round(total_proj[m]))
        headers.append(f"{label} — {cov_u} / {proj_u}")

    return skus, months, coverage, state, headers


def render_heatmap_png(
    skus: list[str],
    headers: list[str],
    coverage: np.ndarray,
    state: np.ndarray,
):
    """
    Render del heatmap v1.0.1:
      0=INACTIVO (celeste, sin %)
      1=ROJO (0-39)
      2=NARANJA (40-79)
      3=VERDE (80-100)
    """
    if len(skus) == 0 or len(headers) == 0:
        return None

    # Colores (celeste, rojo, naranja, verde)
    cmap = ListedColormap(["#8fd3ff", "#ff6b6b", "#ffa94d", "#69db7c"])

    # tamaño dinámico
    fig_w = max(10, 1.6 * len(headers))
    fig_h = max(6, 0.35 * len(skus))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(state, aspect="auto", cmap=cmap, vmin=0, vmax=3)

    ax.set_yticks(np.arange(len(skus)))
    ax.set_yticklabels(skus, fontsize=9)

    ax.set_xticks(np.arange(len(headers)))
    ax.set_xticklabels(headers, rotation=45, ha="right", fontsize=9)

    # textos: porcentaje sin decimales, vacío si INACTIVO
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j] == 0:
                continue
            pct = coverage[i, j]
            if np.isnan(pct):
                continue
            txt = f"{int(round(pct))}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Heatmap de Cobertura — v1.0.1 (CANÓNICO)", fontsize=14)
    ax.set_xlabel("Mes", fontsize=11)
    ax.set_ylabel("SKU", fontsize=11)

    # grid suave para legibilidad
    ax.set_xticks(np.arange(-.5, len(headers), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(skus), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


st.set_page_config(page_title="IA Operativa — Módulo 2 (FASE 2/3)", layout="wide")
ensure_dirs()

st.title("IA Operativa — Módulo 2: Stock y Compras (FASE 2/3)")
st.caption("F2: validaciones duras. F3: escenarios por RUN (LT/cobertura) + outputs CSV estables (2 decimales). Heatmap: diagnóstico v1.0.1.")

uploaded = st.file_uploader("Subí los archivos (xlsx)", accept_multiple_files=True, type=["xlsx"])

if uploaded:
    names = [u.name for u in uploaded]
    a, b = st.columns(2)

    with a:
        modulo_central_name = st.selectbox("Modulo Central.xlsx", ["(no seleccionado)"] + names, 0)
        proyeccion_name = st.selectbox("PROYECCION.xlsx", ["(no seleccionado)"] + names, 0)

    with b:
        importaciones_name = st.selectbox("Importaciones.xlsx", ["(no seleccionado)"] + names, 0)
        fecha_override = st.date_input("FECHA_CORTE_OVERRIDE (opcional)", value=None)
        fecha_override_iso = fecha_override.isoformat() if fecha_override else None

    st.markdown("### Escenario (por RUN)")
    c1, c2 = st.columns(2)
    with c1:
        lead_time_days = st.number_input(
            "LEAD_TIME_DIAS",
            min_value=0,
            max_value=365,
            value=int(LEAD_TIME_DEFAULT_DAYS),
            step=1,
        )
    with c2:
        cobertura_days = st.number_input(
            "COBERTURA_DIAS",
            min_value=1,
            max_value=120,
            value=int(COVERAGE_DEFAULT_DAYS),
            step=1,
        )

    fecha_corte_preview = fecha_override_iso or date.today().isoformat()
    st.info(
        f"Escenario efectivo → FECHA_CORTE: {fecha_corte_preview} | "
        f"LEAD_TIME_DIAS: {int(lead_time_days)} | COBERTURA_DIAS: {int(cobertura_days)}"
    )

    selected = [modulo_central_name, proyeccion_name, importaciones_name]
    can_run_files = "(no seleccionado)" not in selected and len(set(selected)) == 3

    valid_params = (lead_time_days is not None and cobertura_days is not None
                    and int(lead_time_days) >= 0 and int(cobertura_days) >= 1)

    can_run = can_run_files and valid_params

    mode = st.radio(
        "Modo de ejecución",
        [
            "FASE 2 (solo validaciones)",
            "FASE 3.1 (simulación baseline sin compras)",
            "FASE 3.2 (plan de compra con escenario)",
        ],
        index=2,
        horizontal=True,
        disabled=not can_run_files
    )

    st.markdown("### Heatmap de Cobertura (v1.0.1 CANÓNICO)")
    show_heatmap = st.checkbox("Mostrar Heatmap", value=True)
    order_mode = st.selectbox(
        "Orden de SKUs (UI)",
        ["Alfabético", "Criticidad (menor cobertura promedio primero)"],
        index=0
    )

    if not can_run_files:
        st.warning("Mapeo incompleto o repetido. Corregí para habilitar RUN.")
    if can_run_files and not valid_params:
        st.error("Parámetros inválidos. Ajustá LEAD_TIME_DIAS y COBERTURA_DIAS para continuar.")

    if st.button("🚀 RUN", type="primary", disabled=not can_run):
        created_at = now_ts()
        run_id = make_run_id(created_at)
        run_path = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_path, exist_ok=False)

        outputs_dir = os.path.join(run_path, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        meta = save_uploaded_files(run_path, uploaded)
        run_log = build_run_log_base(
            run_id, created_at, meta, fecha_override_iso,
            int(lead_time_days), int(cobertura_days)
        )

        name_to_path = {m["original_name"]: m["stored_path"] for m in meta}
        modulo_central_path = name_to_path[modulo_central_name]
        proyeccion_path = name_to_path[proyeccion_name]
        importaciones_path = name_to_path[importaciones_name]

        run_log["PARAMS_EFECTIVOS"] = {
            "MODULO_CENTRAL_PATH": modulo_central_path,
            "PROYECCION_PATH": proyeccion_path,
            "IMPORTACIONES_PATH": importaciones_path,
        }

        validation_report = {
            "FECHA_CORTE_EFECTIVA": run_log["FECHA_CORTE_EFECTIVA"],
            "VALIDATIONS": [],
            "NOTICES": [],
            "MONTH_COLUMNS_MAP": {},
        }

        heatmap_png = None

        try:
            fecha_corte_efectiva = datetime.fromisoformat(run_log["FECHA_CORTE_EFECTIVA"]).date()
            lt = int(run_log["PARAMETROS_RUN"]["LEAD_TIME_DIAS"])
            cov = int(run_log["PARAMETROS_RUN"]["COBERTURA_DIAS"])

            # ===== FASE 2 =====
            stock_df, mtd_df = read_stock_and_mtd(modulo_central_path)
            validate_mtd_month(mtd_df, fecha_corte_efectiva, modulo_central_path)
            proj_df, month_map, proj_notices = read_and_validate_projection(proyeccion_path, fecha_corte_efectiva)
            transit_df = read_and_validate_transit(importaciones_path)

            run_log["COUNTS"] = {
                "STOCK_ROWS": int(len(stock_df)),
                "MTD_ROWS": int(len(mtd_df)),
                "PROJ_ROWS": int(len(proj_df)),
                "TRANSIT_ROWS": int(len(transit_df)),
                "PROJ_MONTH_COLS": int(len(month_map)),
            }

            validation_report["MONTH_COLUMNS_MAP"] = to_json_safe_month_map(month_map)
            validation_report["NOTICES"].extend(issues_to_dict(proj_notices))
            run_log["STATUS"] = "OK_F2"

            # ===== Heatmap v1.0.1 (NO altera F3) =====
            if show_heatmap:
                skus, months, coverage, state, headers = build_heatmap_cobertura_v101(
                    proj_df=proj_df,
                    month_map=month_map,
                    stock_df=stock_df,
                    transit_df=transit_df,
                    order_mode=order_mode,
                )
                heatmap_png = render_heatmap_png(skus, headers, coverage, state)
                if heatmap_png is not None:
                    # guardado en outputs (opcional, no afecta F3)
                    with open(os.path.join(outputs_dir, "heatmap_cobertura_v1.0.1.png"), "wb") as f:
                        f.write(heatmap_png)

            # ===== F3.1 =====
            sim_df = None
            if mode.startswith("FASE 3.1") or mode.startswith("FASE 3.2"):
                sim_df, kpis_1, f3_notices_1 = simulate_f3_1_baseline(
                    stock_df=stock_df,
                    mtd_df=mtd_df,
                    proj_df=proj_df,
                    transit_df=transit_df,
                    proyeccion_path=proyeccion_path,
                    modulo_central_path=modulo_central_path,
                    importaciones_path=importaciones_path,
                    fecha_corte=fecha_corte_efectiva,
                    lead_time_days=lt,
                    cobertura_days=cov,
                )
                validation_report["NOTICES"].extend(issues_to_dict(f3_notices_1))

                if sim_df is not None and len(sim_df) > 0:
                    write_csv_f3(os.path.join(outputs_dir, "simulation_daily.csv"), sim_df)

                write_json(os.path.join(outputs_dir, "kpis_f3_1.json"), kpis_1)
                run_log["F3"]["KPIS_F3_1"] = kpis_1
                run_log["F3"]["STATUS"] = "OK_F3_1"

            # ===== F3.2 =====
            if mode.startswith("FASE 3.2"):
                plan_df, kpis_2, f3_notices_2 = plan_purchase_f3_2_scenario(
                    simulation_df=sim_df,
                    fecha_corte=fecha_corte_efectiva,
                    proyeccion_path=proyeccion_path,
                    lead_time_days=lt,
                    cobertura_days=cov,
                )
                validation_report["NOTICES"].extend(issues_to_dict(f3_notices_2))

                if plan_df is not None and len(plan_df) > 0:
                    write_csv_f3(os.path.join(outputs_dir, "purchase_plan.csv"), plan_df)

                write_json(os.path.join(outputs_dir, "kpis_f3_2.json"), kpis_2)
                run_log["F3"]["KPIS_F3_2"] = kpis_2
                run_log["F3"]["STATUS"] = "OK_F3_2"

        except HardValidationError as he:
            errs = issues_to_dict(getattr(he, "issues", []))
            validation_report["VALIDATIONS"] = errs
            run_log["VALIDATIONS"] = errs
            run_log["STATUS"] = "ERROR_F2"
            run_log["F3"]["STATUS"] = "SKIPPED_DUE_TO_F2_ERROR"

        except Exception as e:
            t = tech_issue("TECH_UNEXPECTED", str(e))
            validation_report["VALIDATIONS"] = [t]
            run_log["VALIDATIONS"] = [t]
            run_log["STATUS"] = "ERROR_F2"
            run_log["F3"]["STATUS"] = "SKIPPED_DUE_TO_TECH_ERROR"

        write_json(os.path.join(run_path, "validation_report.json"), validation_report)
        write_json(os.path.join(run_path, "run_log.json"), run_log)

        st.success(f"RUN: {run_id} — {run_log['STATUS']} — F3: {run_log['F3']['STATUS']}")

        # Mostrar heatmap + descarga PNG
        if heatmap_png is not None:
            st.image(heatmap_png, caption="Heatmap de Cobertura — v1.0.1 (CANÓNICO)", use_container_width=True)
            st.download_button(
                "⬇️ Descargar PNG (Heatmap)",
                data=heatmap_png,
                file_name=f"{run_id}_heatmap_cobertura_v1.0.1.png",
                mime="image/png",
            )

        st.json(run_log)

        st.download_button(
            "⬇️ Descargar ZIP",
            data=make_zip_bytes(run_path),
            file_name=f"{run_id}.zip",
            mime="application/zip",
        )

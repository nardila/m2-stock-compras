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
from PIL import Image  # <- FIX: render robusto

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
        float_format="%.2f",
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
        "NOTES": "F2: validaciones duras. F3: escenarios por RUN + outputs estables. Heatmap: diagnóstico v1.0.1.",
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
# Heatmap Service Level mensual (CANÓNICO)
# Fuente única: outputs/simulation_daily.csv
#
# Para cada SKU–Mes:
#   DEMANDA_MES   = sum(demand)
#   CUBIERTO_MES  = sum(fulfilled)
#   SERVICE_LEVEL = CUBIERTO_MES / DEMANDA_MES
#
# Caso especial:
#   Si DEMANDA_MES = 0 -> SKU INACTIVO ese mes (celeste)
#
# Visual:
#   80%–100%  -> verde
#   40%–79%   -> naranja
#   0%–39%    -> rojo
#   INACTIVO  -> celeste
# -----------------------------

def _spanish_month_label(month_start: date, include_year: bool) -> str:
    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    base = months[month_start.month - 1]
    return f"{base} {month_start.year}" if include_year else base


def build_heatmap_service_level_from_simulation_csv(sim_csv_path: str, order_mode: str):
    """Construye el heatmap desde outputs/simulation_daily.csv (fuente única)."""
    if not os.path.exists(sim_csv_path):
        return [], [], np.array([[]]), np.array([[]], dtype=int)

    df = pd.read_csv(sim_csv_path)
    required = {"date", "SKU", "demand", "fulfilled"}
    if not required.issubset(set(df.columns)):
        return [], [], np.array([[]]), np.array([[]], dtype=int)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    df["fulfilled"] = pd.to_numeric(df["fulfilled"], errors="coerce").fillna(0.0)

    # Mes calendario (primer día del mes)
    df["MONTH"] = df["date"].dt.to_period("M").dt.to_timestamp().dt.date

    # Agregación canónica (mensual)
    agg = df.groupby(["SKU", "MONTH"], as_index=False).agg(
        DEMANDA_MES=("demand", "sum"),
        CUBIERTO_MES=("fulfilled", "sum"),
    )

    # Universo de meses (ordenado)
    months = sorted(agg["MONTH"].unique().tolist())
    if not months:
        return [], [], np.array([[]]), np.array([[]], dtype=int)

    # Incluir solo SKUs con alguna demanda en el horizonte
    total_demand_by_sku = agg.groupby("SKU")["DEMANDA_MES"].sum()
    skus = sorted([sku for sku, td in total_demand_by_sku.items() if float(td) > 0.0])
    if not skus:
        return [], months, np.array([[]]), np.array([[]], dtype=int)

    sku_index = {s: i for i, s in enumerate(skus)}
    month_index = {m: j for j, m in enumerate(months)}

    S, K = len(skus), len(months)

    # pct: porcentaje 0-100; NaN para INACTIVO
    pct = np.full((S, K), np.nan, dtype=float)
    # state: 0 INACTIVO, 1 rojo, 2 naranja, 3 verde
    state = np.zeros((S, K), dtype=int)

    for _, r in agg.iterrows():
        sku = r["SKU"]
        if sku not in sku_index:
            continue
        m = r["MONTH"]
        i = sku_index[sku]
        j = month_index[m]

        dem = float(r["DEMANDA_MES"])
        ful = float(r["CUBIERTO_MES"])

        if dem <= 0.0:
            state[i, j] = 0
            continue

        sl = ful / dem
        sl = max(0.0, min(1.0, sl))
        v = sl * 100.0
        pct[i, j] = v

        if v >= 80.0:
            state[i, j] = 3
        elif v >= 40.0:
            state[i, j] = 2
        else:
            state[i, j] = 1

    # Orden (solo UI)
    if order_mode.startswith("Criticidad"):
        avg_sl = []
        for i, sku in enumerate(skus):
            vals = pct[i, :]
            vals = vals[~np.isnan(vals)]
            avg = float(np.mean(vals)) if len(vals) else 0.0
            avg_sl.append((avg, sku, i))
        avg_sl.sort(key=lambda x: (x[0], x[1]))
        order_idx = [x[2] for x in avg_sl]
        skus = [skus[i] for i in order_idx]
        pct = pct[order_idx, :]
        state = state[order_idx, :]

    return skus, months, pct, state


def render_heatmap_service_level_png(
    skus: list[str],
    months: list[date],
    pct: np.ndarray,
    state: np.ndarray,
) -> bytes | None:
    if not skus or not months:
        return None

    years = {m.year for m in months}
    include_year = len(years) > 1
    xlabels = [_spanish_month_label(m, include_year=include_year) for m in months]

    # 0=INACTIVO(celeste), 1=rojo, 2=naranja, 3=verde
    cmap = ListedColormap(["#8fd3ff", "#ff6b6b", "#ffa94d", "#69db7c"])

    fig_w = max(10, 1.2 * len(xlabels))
    fig_h = max(6, 0.35 * len(skus))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(state, aspect="auto", cmap=cmap, vmin=0, vmax=3)

    ax.set_yticks(np.arange(len(skus)))
    ax.set_yticklabels(skus, fontsize=9)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)

    # Texto en celdas: solo porcentaje (sin decimales). No mostrar texto en INACTIVO.
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j] == 0:
                continue
            v = pct[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{int(round(v))}%", ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Heatmap Service Level mensual", fontsize=14)
    ax.set_xlabel("Mes", fontsize=11)
    ax.set_ylabel("SKU", fontsize=11)

    ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(skus), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)

    png_bytes = buf.getvalue()
    if not (isinstance(png_bytes, (bytes, bytearray)) and len(png_bytes) > 8 and png_bytes[:4] == b"\x89PNG"):
        return None
    return png_bytes


st.set_page_config(page_title="IA Operativa — Módulo 2 (FASE 2/3)", layout="wide")
ensure_dirs()

st.title("IA Operativa — Módulo 2: Stock y Compras (FASE 2/3)")
st.caption("F2: validaciones duras. F3: escenarios por RUN + outputs estables. Heatmap: diagnóstico v1.0.1.")

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
        lead_time_days = st.number_input("LEAD_TIME_DIAS", 0, 365, int(LEAD_TIME_DEFAULT_DAYS), 1)
    with c2:
        cobertura_days = st.number_input("COBERTURA_DIAS", 1, 120, int(COVERAGE_DEFAULT_DAYS), 1)

    fecha_corte_preview = fecha_override_iso or date.today().isoformat()
    st.info(
        f"Escenario efectivo → FECHA_CORTE: {fecha_corte_preview} | "
        f"LEAD_TIME_DIAS: {int(lead_time_days)} | COBERTURA_DIAS: {int(cobertura_days)}"
    )

    selected = [modulo_central_name, proyeccion_name, importaciones_name]
    can_run_files = "(no seleccionado)" not in selected and len(set(selected)) == 3
    valid_params = int(lead_time_days) >= 0 and int(cobertura_days) >= 1
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

    st.markdown("### Heatmap Service Level mensual (CANÓNICO)")
    show_heatmap = st.checkbox("Mostrar Heatmap", value=True)
    order_mode = st.selectbox("Orden de SKUs (UI)", ["Alfabético", "Criticidad (menor cobertura promedio primero)"], 0)

    if st.button("🚀 RUN", type="primary", disabled=not can_run):
        created_at = now_ts()
        run_id = make_run_id(created_at)
        run_path = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_path, exist_ok=False)

        outputs_dir = os.path.join(run_path, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        meta = save_uploaded_files(run_path, uploaded)
        run_log = build_run_log_base(run_id, created_at, meta, fecha_override_iso, int(lead_time_days), int(cobertura_days))

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

            # Heatmap: se calcula exclusivamente desde outputs/simulation_daily.csv (fuente única).
            # Nota: si el modo es FASE 2, no existe simulation_daily.csv, por lo tanto no se genera heatmap.
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

                    # Generar heatmap (service level mensual) desde el CSV recién exportado
                    if show_heatmap:
                        sim_csv_path = os.path.join(outputs_dir, "simulation_daily.csv")
                        skus, months, pct, state = build_heatmap_service_level_from_simulation_csv(sim_csv_path, order_mode)
                        heatmap_png = render_heatmap_service_level_png(skus, months, pct, state)
                        if heatmap_png is not None:
                            with open(os.path.join(outputs_dir, "heatmap_service_level.png"), "wb") as f:
                                f.write(heatmap_png)

                write_json(os.path.join(outputs_dir, "kpis_f3_1.json"), kpis_1)
                run_log["F3"]["KPIS_F3_1"] = kpis_1
                run_log["F3"]["STATUS"] = "OK_F3_1"

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

        # ---- FIX definitivo: render con PIL para evitar crash de Streamlit ----
        if show_heatmap:
            if isinstance(heatmap_png, (bytes, bytearray)) and len(heatmap_png) > 8 and heatmap_png[:4] == b"\x89PNG":
                try:
                    img = Image.open(io.BytesIO(heatmap_png))
                    st.image(img, caption="Heatmap Service Level mensual (CANÓNICO)", use_container_width=True)
                    st.download_button(
                        "⬇️ Descargar PNG (Heatmap)",
                        data=heatmap_png,
                        file_name=f"{run_id}_heatmap_service_level.png",
                        mime="image/png",
                    )
                except Exception:
                    st.warning("Heatmap generado pero no pudo renderizarse en la UI. Descargalo desde el ZIP (outputs/heatmap_service_level.png).")
            else:
                st.warning("Heatmap no generado (PNG vacío o inválido). Revisar que exista simulation_daily.csv (ejecutar FASE 3) y que haya demanda en el horizonte.")

        st.json(run_log)

        st.download_button(
            "⬇️ Descargar ZIP",
            data=make_zip_bytes(run_path),
            file_name=f"{run_id}.zip",
            mime="application/zip",
        )

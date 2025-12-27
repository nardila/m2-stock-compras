import os
import io
import json
import zipfile
from datetime import datetime, date
import hashlib
import secrets
import streamlit as st
import pandas as pd

from engine_f2 import (
    HardValidationError,
    ValidationIssue,
    read_stock_and_mtd,
    validate_mtd_month,
    read_and_validate_projection,
    read_and_validate_transit,
)

from engine_f3 import simulate_f3_1_baseline

RUNS_DIR = "runs"


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


def write_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False, encoding="utf-8")


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


def build_run_log_base(run_id: str, created_at: datetime, input_files_meta, fecha_corte_override_iso: str | None):
    fecha_corte_default = created_at.date().isoformat()
    fecha_corte_efectiva = fecha_corte_override_iso or fecha_corte_default
    return {
        "RUN_ID": run_id,
        "CREATED_AT": created_at.isoformat(timespec="seconds"),
        "FECHA_CORTE_DEFAULT": fecha_corte_default,
        "FECHA_CORTE_OVERRIDE": fecha_corte_override_iso,
        "FECHA_CORTE_EFECTIVA": fecha_corte_efectiva,
        "OVERRIDE_ACTIVO": bool(fecha_corte_override_iso),
        "INPUT_FILES": input_files_meta,
        "STATUS": None,
        "VALIDATIONS": [],
        "PARAMS_EFECTIVOS": {},
        "COUNTS": {},
        "NOTES": "FASE 2/3: Validaciones duras + F3.1 simulación baseline (sin compras).",
        "F3": {
            "STATUS": None,
            "KPIS": {},
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


st.set_page_config(page_title="IA Operativa — Módulo 2 (FASE 2/3)", layout="wide")
ensure_dirs()

st.title("IA Operativa — Módulo 2: Stock y Compras (FASE 2/3)")
st.caption("F2: validaciones duras. F3.1: simulación baseline diaria con decimales (sin compras).")

uploaded = st.file_uploader("Subí los archivos (xlsx)", accept_multiple_files=True, type=["xlsx"])

if uploaded:
    names = [u.name for u in uploaded]
    a, b = st.columns(2)

    with a:
        modulo_central_name = st.selectbox("Modulo Central.xlsx", ["(no seleccionado)"] + names, 0)
        proyeccion_name = st.selectbox("PROYECCION - SKU  MAYO 2026.xlsx", ["(no seleccionado)"] + names, 0)

    with b:
        importaciones_name = st.selectbox("Importaciones (1).xlsx", ["(no seleccionado)"] + names, 0)
        fecha_override = st.date_input("FECHA_CORTE_OVERRIDE (opcional)", value=None)
        fecha_override_iso = fecha_override.isoformat() if fecha_override else None

    selected = [modulo_central_name, proyeccion_name, importaciones_name]
    can_run = "(no seleccionado)" not in selected and len(set(selected)) == 3

    mode = st.radio(
        "Modo de ejecución",
        ["FASE 2 (solo validaciones)", "FASE 3.1 (simulación baseline sin compras)"],
        index=1,
        horizontal=True,
        disabled=not can_run
    )

    if not can_run:
        st.warning("Mapeo incompleto o repetido. Corregí para habilitar RUN.")

    if st.button("🚀 RUN", type="primary", disabled=not can_run):
        created_at = now_ts()
        run_id = make_run_id(created_at)
        run_path = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_path, exist_ok=False)

        outputs_dir = os.path.join(run_path, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        meta = save_uploaded_files(run_path, uploaded)
        run_log = build_run_log_base(run_id, created_at, meta, fecha_override_iso)

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

        try:
            fecha_corte_efectiva = datetime.fromisoformat(run_log["FECHA_CORTE_EFECTIVA"]).date()

            # ===== FASE 2 (siempre) =====
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

            # ===== FASE 3.1 =====
            if mode.startswith("FASE 3.1"):
                sim_df, kpis, f3_notices = simulate_f3_1_baseline(
                    stock_df=stock_df,
                    mtd_df=mtd_df,
                    proj_df=proj_df,
                    transit_df=transit_df,
                    proyeccion_path=proyeccion_path,
                    modulo_central_path=modulo_central_path,
                    importaciones_path=importaciones_path,
                    fecha_corte=fecha_corte_efectiva,
                )

                # NOTICES F3 (mismo contrato)
                validation_report["NOTICES"].extend(issues_to_dict(f3_notices))

                # Outputs
                if sim_df is not None and len(sim_df) > 0:
                    write_csv(os.path.join(outputs_dir, "simulation_daily.csv"), sim_df)

                write_json(os.path.join(outputs_dir, "kpis.json"), kpis)

                run_log["F3"]["STATUS"] = "OK_F3_1"
                run_log["F3"]["KPIS"] = kpis

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

        # Persistir
        write_json(os.path.join(run_path, "validation_report.json"), validation_report)
        write_json(os.path.join(run_path, "run_log.json"), run_log)

        st.success(f"RUN: {run_id} — {run_log['STATUS']} — F3: {run_log['F3']['STATUS']}")
        st.json(run_log)

        st.download_button(
            "⬇️ Descargar ZIP",
            data=make_zip_bytes(run_path),
            file_name=f"{run_id}.zip",
            mime="application/zip",
        )

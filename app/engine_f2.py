import os
import io
import json
import zipfile
from datetime import datetime, date
import hashlib
import secrets
import streamlit as st

from engine_f2 import (
    HardValidationError,
    ValidationIssue,
    read_stock_and_mtd,
    validate_mtd_month,
    read_and_validate_projection,
    read_and_validate_transit,
)

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
        "ERRORS": [],
        "PARAMS_EFECTIVOS": {},
        "COUNTS": {},
        "NOTES": "FASE 2: Lectura estricta y validaciones duras. Sin simulación ni outputs de negocio.",
    }


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    # Garantiza el contrato de trazabilidad en JSON
    out = []
    for i in issues:
        out.append({
            "file": i.file,
            "sheet": i.sheet,
            "column": i.column,
            "bad_rows": i.bad_rows,
            "bad_count": i.bad_count,
            "code": i.code,
            "message": i.message,
            "type": i.type,  # DATA_ERROR / TECH_ERROR
        })
    return out


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def to_json_safe_month_map(month_map: dict) -> dict:
    """
    Evita keys datetime en JSON:
    - key: YYYY-MM (string)
    - value: date iso + header original string
    """
    safe = {}
    for col_header, d in month_map.items():
        k = month_key(d)
        safe[k] = {
            "date": d.isoformat(),
            "source_col_header": str(col_header)
        }
    return safe


def make_tech_issue(file: str, sheet: str, column: str | None, code: str, message: str) -> dict:
    # Incluso TECH_ERROR cumple el contrato completo (sin inferencias).
    return {
        "file": file,
        "sheet": sheet,
        "column": column,
        "bad_rows": [],
        "bad_count": 0,
        "code": code,
        "message": message,
        "type": "TECH_ERROR",
    }


st.set_page_config(page_title="IA Operativa — Módulo 2 (FASE 2)", layout="wide")
ensure_dirs()

st.title("IA Operativa — Módulo 2: Stock y Compras (FASE 2)")
st.caption("FASE 2: Lectura estricta de inputs + Validaciones duras (falla total) con trazabilidad completa.")

uploaded = st.file_uploader(
    "Subí los archivos del Módulo 2 (xlsx). En FASE 2 se validan estrictamente.",
    accept_multiple_files=True,
    type=["xlsx"]
)

if uploaded and len(uploaded) > 0:
    st.subheader("Mapeo de archivos (obligatorio en FASE 2)")
    names = [u.name for u in uploaded]

    colA, colB = st.columns(2)

    with colA:
        modulo_central_name = st.selectbox(
            "Seleccioná el archivo que corresponde a **Modulo Central.xlsx**",
            options=["(no seleccionado)"] + names,
            index=0
        )
        proyeccion_name = st.selectbox(
            "Seleccioná el archivo que corresponde a **PROYECCION - SKU  MAYO 2026.xlsx**",
            options=["(no seleccionado)"] + names,
            index=0
        )

    with colB:
        importaciones_name = st.selectbox(
            "Seleccioná el archivo que corresponde a **Importaciones (1).xlsx**",
            options=["(no seleccionado)"] + names,
            index=0
        )
        fecha_override = st.date_input(
            "FECHA_CORTE_OVERRIDE (opcional)",
            value=None
        )
        fecha_override_iso = fecha_override.isoformat() if fecha_override else None

    selected = [modulo_central_name, proyeccion_name, importaciones_name]
    if "(no seleccionado)" in selected:
        st.warning("Seleccioná los 3 archivos requeridos para habilitar RUN.")
        can_run = False
    elif len(set(selected)) != 3:
        st.error("No podés asignar el mismo archivo a más de un input. Corregí el mapeo.")
        can_run = False
    else:
        can_run = True

    run_clicked = st.button("🚀 RUN (FASE 2)", type="primary", disabled=not can_run)

    if run_clicked:
        created_at = now_ts()
        run_id = make_run_id(created_at)
        run_path = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_path, exist_ok=False)

        meta = save_uploaded_files(run_path, uploaded)
        run_log = build_run_log_base(run_id, created_at, meta, fecha_override_iso)

        name_to_path = {m["original_name"]: m["stored_path"] for m in meta}
        modulo_central_path = name_to_path[modulo_central_name]
        proyeccion_path = name_to_path[proyeccion_name]
        importaciones_path = name_to_path[importaciones_name]

        run_log["PARAMS_EFECTIVOS"]["MODULO_CENTRAL_PATH"] = modulo_central_path
        run_log["PARAMS_EFECTIVOS"]["PROYECCION_PATH"] = proyeccion_path
        run_log["PARAMS_EFECTIVOS"]["IMPORTACIONES_PATH"] = importaciones_path

        all_issues: list[ValidationIssue] = []
        validation_report = {
            "FECHA_CORTE_EFECTIVA": run_log["FECHA_CORTE_EFECTIVA"],
            "VALIDATIONS": [],
            "MONTH_COLUMNS_MAP": {},
        }

        try:
            stock_df, mtd_df, issues = read_stock_and_mtd(modulo_central_path)
            all_issues.extend(issues)

            fecha_corte_efectiva = datetime.fromisoformat(run_log["FECHA_CORTE_EFECTIVA"]).date()
            all_issues.extend(validate_mtd_month(mtd_df, fecha_corte_efectiva, modulo_central_path))

            proj_df, month_map, proj_issues = read_and_validate_projection(proyeccion_path, fecha_corte_efectiva)
            all_issues.extend(proj_issues)

            transit_df, impo_issues = read_and_validate_transit(importaciones_path)
            all_issues.extend(impo_issues)

            run_log["COUNTS"] = {
                "STOCK_ROWS": int(len(stock_df)),
                "MTD_ROWS": int(len(mtd_df)),
                "PROJ_ROWS": int(len(proj_df)),
                "TRANSIT_ROWS": int(len(transit_df)),
                "PROJ_MONTH_COLS": int(len(month_map)),
            }

            validation_report["VALIDATIONS"] = issues_to_dict(all_issues)
            validation_report["MONTH_COLUMNS_MAP"] = to_json_safe_month_map(month_map)

            run_log["VALIDATIONS"] = validation_report["VALIDATIONS"]
            run_log["STATUS"] = "OK_F2"

        except HardValidationError as he:
            # Falla total, pero con trazabilidad completa.
            run_log["STATUS"] = "ERROR_F2"
            run_log["VALIDATIONS"] = issues_to_dict(getattr(he, "issues", []))
            run_log["ERRORS"] = ["Validaciones duras fallaron. Falla total (no se genera nada parcial)."]

            validation_report["VALIDATIONS"] = run_log["VALIDATIONS"]

        except Exception as e:
            # TECH_ERROR estructurado (sin genéricos)
            run_log["STATUS"] = "ERROR_F2"
            tech = make_tech_issue(
                file="(runtime)",
                sheet="(runtime)",
                column=None,
                code="TECH_UNEXPECTED",
                message=str(e),
            )
            run_log["VALIDATIONS"] = [tech]
            run_log["ERRORS"] = ["Error técnico inesperado. Ver VALIDATIONS para trazabilidad."]
            validation_report["VALIDATIONS"] = [tech]

        # Persistir reportes
        write_json(os.path.join(run_path, "validation_report.json"), validation_report)
        write_json(os.path.join(run_path, "run_log.json"), run_log)

        st.success(f"RUN generado: {run_id} — STATUS={run_log['STATUS']}")
        st.json(run_log)

        zip_bytes = make_zip_bytes(run_path)
        st.download_button(
            label="⬇️ Descargar ZIP del RUN (incluye run_log + validation_report + inputs)",
            data=zip_bytes,
            file_name=f"{run_id}.zip",
            mime="application/zip"
        )

st.divider()
st.subheader("Historial local (runs/)")
runs = sorted([d for d in os.listdir(RUNS_DIR) if d.startswith("RUN_")], reverse=True)
if not runs:
    st.write("Todavía no hay RUNs.")
else:
    st.write(f"RUNs detectados: {len(runs)}")
    st.code("\n".join(runs[:20]))

import streamlit as st
from datetime import datetime
import json
import zipfile
from pathlib import Path
import secrets

st.set_page_config(page_title="Módulo 2 – Stock y Compras")

st.title("Módulo 2 – Stock y Compras")
st.caption("FASE 1 – Carga de archivos y generación de RUN")

def make_run_id():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = secrets.token_hex(3)
    return f"RUN_{ts}_{rnd}"

def save_file(uploaded_file, path):
    path.write_bytes(uploaded_file.getbuffer())

def zip_folder(folder, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(folder))

st.markdown("### Subí los archivos")

stock = st.file_uploader("Stock actual", type=["xlsx", "csv"])
incoming = st.file_uploader("Próximos ingresos", type=["xlsx", "csv"])
forecast = st.file_uploader("Proyección de ventas", type=["xlsx", "csv"])

if st.button("RUN"):
    if not all([stock, incoming, forecast]):
        st.error("Tenés que subir los 3 archivos.")
    else:
        run_id = make_run_id()
        base = Path("runs") / run_id
        inputs = base / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)

        save_file(stock, inputs / stock.name)
        save_file(incoming, inputs / incoming.name)
        save_file(forecast, inputs / forecast.name)

        run_log = {
            "RUN_ID": run_id,
            "STATUS": "OK_F1",
            "TIMESTAMP": datetime.now().isoformat()
        }

        (base / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")

        zip_path = base / f"{run_id}.zip"
        zip_folder(base, zip_path)

        st.success(f"Corrida creada: {run_id}")
        st.download_button(
            "Descargar ZIP",
            zip_path.read_bytes(),
            file_name=f"{run_id}.zip"
        )


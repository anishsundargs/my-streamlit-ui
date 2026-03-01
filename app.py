import os
import re
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


APP_NAME = "Passion Project UI (Mock Backend + Real Backend Hook)"


# ----------------------------
# Helpers
# ----------------------------
def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name or "uploaded_file"


def read_uploaded_as_df(uploaded_file) -> pd.DataFrame:
    """Best-effort parse CSV / TSV / Excel into a dataframe."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".csv"]:
        return pd.read_csv(uploaded_file)
    if suffix in [".tsv", ".txt"]:
        return pd.read_csv(uploaded_file, sep="\t")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {suffix} (upload .csv, .tsv, .xlsx)")


def mock_predict(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """
    Fake prediction generator so your UI works today.
    - If df has an 'allele' column, we use it.
    - Otherwise we synthesize allele names.
    - We produce a 'score' column in [0,1].
    """
    out = df.copy()

    # Ensure an allele column exists (common for immunology pipelines)
    if "allele" not in out.columns:
        out["allele"] = [f"HLA-A*{i%30:02d}:01" for i in range(len(out))]

    # Create deterministic-ish scores without numpy
    # (keeps it dependency-light and reproducible)
    scores = []
    for i, a in enumerate(out["allele"].astype(str).tolist()):
        # hash-like score in [0,1]
        h = sum((ord(c) * (i + 1)) for c in a) % 1000
        scores.append(h / 1000.0)
    out["score"] = scores

    out.to_csv(out_csv, index=False)
    return out


def try_run_real_backend(
    input_path: Path,
    output_dir: Path,
    command_template: str,
    timeout_sec: int = 1800,
) -> subprocess.CompletedProcess | None:
    """
    Attempts to run a real backend command.
    command_template can include:
      {input} -> input_path
      {outdir} -> output_dir
    Example:
      "python -m predict_pl data.file={input} out_dir={outdir}"

    Returns CompletedProcess or None if command_template is empty.
    """
    if not command_template.strip():
        return None

    cmd = command_template.format(input=str(input_path), outdir=str(output_dir)).strip()

    # Basic safety: run as a list for subprocess, avoid shell=True
    # Split on spaces but preserve quoted segments:
    import shlex
    args = shlex.split(cmd)

    return subprocess.run(
        args,
        cwd=str(output_dir),          # run inside output folder
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def plot_score_histogram(df: pd.DataFrame):
    if "score" not in df.columns:
        st.warning("No 'score' column to plot.")
        return
    fig = plt.figure()
    plt.hist(df["score"].dropna().astype(float), bins=20)
    plt.xlabel("score")
    plt.ylabel("count")
    st.pyplot(fig)


def plot_mean_score_by_allele(df: pd.DataFrame, top_n: int = 15):
    if "allele" not in df.columns or "score" not in df.columns:
        st.warning("Need 'allele' and 'score' columns for allele plot.")
        return

    tmp = df.copy()
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
    grouped = (
        tmp.groupby("allele", dropna=True)["score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig = plt.figure()
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("mean score")
    plt.tight_layout()
    st.pyplot(fig)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

st.markdown(
    """
This app works **now** with a **mock backend** and has a **drop-in hook** for your friend's real pipeline later.
- Upload a CSV/TSV/XLSX
- Preview it
- Click **Run**
- Get a `preds.csv` + plots + downloads
"""
)

with st.sidebar:
    st.header("Backend mode")
    mode = st.radio(
        "Choose backend",
        options=["Mock (works now)", "Real command (later)"],
        index=0,
    )

    st.caption("When you get your friend's repo, switch to **Real command** and paste his run command.")
    command_template = ""
    if mode == "Real command (later)":
        command_template = st.text_area(
            "Command template (no shell). Use {input} and {outdir}",
            value="python -m predict_pl data.file={input} out_dir={outdir}",
            height=90,
        )
        timeout_sec = st.number_input("Timeout (seconds)", min_value=30, max_value=7200, value=1800, step=30)
    else:
        timeout_sec = 1800

    st.header("UI options")
    top_n = st.slider("Top alleles to display", min_value=5, max_value=30, value=15, step=1)


uploaded = st.file_uploader("Upload input file (.csv, .tsv, .xlsx)", type=["csv", "tsv", "txt", "xlsx", "xls"])

if uploaded is None:
    st.stop()

# Load input data
try:
    df_in = read_uploaded_as_df(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Input preview")
    st.write(f"Rows: **{len(df_in)}** | Columns: **{len(df_in.columns)}**")
    st.dataframe(df_in.head(200), use_container_width=True)

with right:
    st.subheader("Run")
    st.write("This will produce a `preds.csv` output and plots.")

    if st.button("Run prediction", type="primary"):
        with st.spinner("Running..."):
            # Work directory for this run
            run_dir = Path(tempfile.mkdtemp(prefix="ui_run_"))
            input_name = safe_filename(uploaded.name)
            input_path = run_dir / input_name

            # Save uploaded file to disk so a real backend can read it
            with open(input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            out_dir = run_dir / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)

            preds_csv = out_dir / "preds.csv"

            logs = {"mode": mode, "run_dir": str(run_dir), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            backend_stdout = ""
            backend_stderr = ""
            return_code = None

            if mode == "Mock (works now)":
                df_out = mock_predict(df_in, preds_csv)
                return_code = 0
            else:
                try:
                    cp = try_run_real_backend(
                        input_path=input_path,
                        output_dir=out_dir,
                        command_template=command_template,
                        timeout_sec=int(timeout_sec),
                    )
                    if cp is None:
                        raise RuntimeError("No command provided.")
                    backend_stdout = cp.stdout
                    backend_stderr = cp.stderr
                    return_code = cp.returncode

                    # If the command didn't create preds.csv, fail loudly
                    if not preds_csv.exists():
                        raise RuntimeError(
                            "Backend ran, but preds.csv was not created in the output directory. "
                            "Fix the command template or ask your friend where outputs are written."
                        )

                    df_out = pd.read_csv(preds_csv)

                except Exception as e:
                    st.error(f"Real backend failed: {e}")
                    if backend_stderr:
                        st.code(backend_stderr)
                    st.stop()

            # Save run metadata
            logs["return_code"] = return_code
            meta_path = out_dir / "run_metadata.json"
            meta_path.write_text(json.dumps(logs, indent=2))

            # Persist in session so UI can show outputs after reruns
            st.session_state["last_out_dir"] = str(out_dir)
            st.session_state["last_preds_csv"] = str(preds_csv)
            st.session_state["last_meta"] = str(meta_path)
            st.session_state["last_stdout"] = backend_stdout
            st.session_state["last_stderr"] = backend_stderr

        st.success("Done.")


# ----------------------------
# Results display
# ----------------------------
if "last_preds_csv" in st.session_state and Path(st.session_state["last_preds_csv"]).exists():
    st.divider()
    st.header("Results")

    preds_csv = Path(st.session_state["last_preds_csv"])
    out_dir = Path(st.session_state["last_out_dir"])
    df_out = pd.read_csv(preds_csv)

    a, b = st.columns([2, 1], gap="large")

    with a:
        st.subheader("preds.csv preview")
        st.write(f"Rows: **{len(df_out)}** | Columns: **{len(df_out.columns)}**")
        st.dataframe(df_out.head(500), use_container_width=True)

        st.subheader("Plots")
        plot_score_histogram(df_out)
        plot_mean_score_by_allele(df_out, top_n=top_n)

    with b:
        st.subheader("Downloads")
        st.download_button(
            label="Download preds.csv",
            data=preds_csv.read_bytes(),
            file_name="preds.csv",
            mime="text/csv",
        )
        meta_path = Path(st.session_state["last_meta"])
        if meta_path.exists():
            st.download_button(
                label="Download run_metadata.json",
                data=meta_path.read_bytes(),
                file_name="run_metadata.json",
                mime="application/json",
            )

        if st.session_state.get("last_stdout"):
            with st.expander("Backend stdout"):
                st.code(st.session_state["last_stdout"])
        if st.session_state.get("last_stderr"):
            with st.expander("Backend stderr"):
                st.code(st.session_state["last_stderr"])

        st.subheader("Output folder")
        st.code(str(out_dir))

        st.caption(
            "Later, when you have the real repo, switch backend mode to 'Real command' and paste the exact command your friend uses."
        )

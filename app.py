"""
app.py  –  Interface Streamlit para o Gerador de Agenda Semanal
===============================================================
Arquivos necessários no repositório GitHub:
    app.py
    gerar_agenda_semanal.py
    requirements.txt
    logo_header_institucional.png   (opcional)
"""

import io
import zipfile
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from gerar_agenda_semanal import (
    carregar_excel,
    filtrar_semana,
    desenhar_card_agenda,
    semana_de,
    slug,
)

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Agenda Semanal – CEAD",
    page_icon="📅",
    layout="centered",
)

# ─── Cabeçalho ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:#0B2A3A;padding:22px 24px 16px;border-radius:8px;margin-bottom:28px">
        <h2 style="color:#fff;margin:0;font-size:1.4rem;font-weight:700">
            📅 Gerador de Agenda Semanal
        </h2>
        <p style="color:#AACCDD;margin:5px 0 0;font-size:0.9rem">
            Licenciatura em Física · CEAD · UFPI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Upload da planilha ───────────────────────────────────────────────────────
st.markdown("#### 1 · Planilha de atividades")
arquivo = st.file_uploader(
    "Selecione o arquivo Excel (.xlsx)",
    type=["xlsx"],
    label_visibility="collapsed",
)

df = None
if arquivo:
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(arquivo.read())
            tmp_path = tmp.name
        df = carregar_excel(tmp_path)
        st.success(f"✅  Planilha carregada com {len(df)} linha(s).")
    except Exception as e:
        st.error(f"Não foi possível ler a planilha: {e}")

# ─── Data da semana ───────────────────────────────────────────────────────────
st.markdown("#### 2 · Semana")
data_ref = st.date_input(
    "Escolha qualquer data dentro da semana desejada",
    value=date.today(),
    format="DD/MM/YYYY",
    label_visibility="collapsed",
)

if data_ref:
    ts = pd.Timestamp(data_ref)
    segunda, domingo = semana_de(ts)
    st.caption(f"Semana: **{segunda.strftime('%d/%m/%Y')}** (segunda) até **{domingo.strftime('%d/%m/%Y')}** (domingo)")

# ─── Botão gerar ──────────────────────────────────────────────────────────────
st.markdown("#### 3 · Gerar")
gerar = st.button(
    "🖼️  Gerar Agenda",
    type="primary",
    disabled=(df is None),
    use_container_width=True,
)

if gerar and df is not None:
    data_ref_ts = pd.Timestamp(data_ref)
    segunda, domingo = semana_de(data_ref_ts)
    label_semana = f"Semana {segunda.strftime('%d/%m')} – {domingo.strftime('%d/%m/%Y')}"

    rec = filtrar_semana(df, data_ref_ts, None, None)

    if rec.empty:
        st.warning("⚠️  Nenhuma atividade encontrada para esta semana na planilha.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        base   = f"agenda_{data_ref_ts.strftime('%Y-%m-%d')}"

        with st.spinner("Gerando imagens..."):
            desenhar_card_agenda(
                rec,
                polo="",
                modulo="",
                label_semana=label_semana,
                aspect="4:5",
                out_png=outdir / f"{base}.png",
                out_pdf=outdir / f"{base}.pdf",
            )

        pngs = sorted(outdir.glob("*.png"))
        pdfs = sorted(outdir.glob("*.pdf"))

        if not pngs:
            st.error("Nenhuma imagem foi gerada. Verifique a planilha.")
            st.stop()

        st.success(f"✅  {len(pngs)} card(s) gerado(s).")

        # Preview dos cards na página
        for png in pngs:
            st.image(str(png), use_container_width=True)

        st.markdown("---")

        nome_base = f"agenda_semanal_{data_ref_ts.strftime('%Y-%m-%d')}"
        col1, col2 = st.columns(2)

        # ── PNG ──
        if len(pngs) == 1:
            png_data  = pngs[0].read_bytes()
            png_nome  = pngs[0].name
            png_mime  = "image/png"
        else:
            buf_png = io.BytesIO()
            with zipfile.ZipFile(buf_png, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in pngs:
                    zf.write(f, arcname=f.name)
            buf_png.seek(0)
            png_data = buf_png.read()
            png_nome = f"{nome_base}_imagens.zip"
            png_mime = "application/zip"

        # ── PDF ──
        if len(pdfs) == 1:
            pdf_data = pdfs[0].read_bytes()
            pdf_nome = pdfs[0].name
            pdf_mime = "application/pdf"
        else:
            buf_pdf = io.BytesIO()
            with zipfile.ZipFile(buf_pdf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in pdfs:
                    zf.write(f, arcname=f.name)
            buf_pdf.seek(0)
            pdf_data = buf_pdf.read()
            pdf_nome = f"{nome_base}_pdfs.zip"
            pdf_mime = "application/zip"

        label_png = "⬇️  Baixar PNG" if len(pngs) == 1 else f"⬇️  Baixar PNGs (ZIP · {len(pngs)} cards)"
        label_pdf = "⬇️  Baixar PDF" if len(pdfs) == 1 else f"⬇️  Baixar PDFs (ZIP · {len(pdfs)} cards)"

        with col1:
            st.download_button(label_png, data=png_data, file_name=png_nome,
                               mime=png_mime, use_container_width=True)
        with col2:
            st.download_button(label_pdf, data=pdf_data, file_name=pdf_nome,
                               mime=pdf_mime, use_container_width=True)

# ─── Rodapé ───────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;color:#9AA4B0;font-size:0.8rem;margin-top:40px'>"
    "CEAD · Universidade Federal do Piauí · cead.ufpi.br"
    "</p>",
    unsafe_allow_html=True,
)

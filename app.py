"""
app.py  –  Interface web (Streamlit) para o Gerador de Agenda Semanal
=====================================================================
Hospede este arquivo junto com gerar_agenda_semanal.py no mesmo
repositório GitHub e publique no Streamlit Cloud.

Arquivos necessários no repositório:
    app.py
    gerar_agenda_semanal.py
    requirements.txt
    logo_header_institucional.png   (opcional)
"""

import io
import zipfile
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Importa as funções do script original — sem nenhuma alteração nele
from gerar_agenda_semanal import (
    carregar_excel,
    filtrar_semana,
    desenhar_card_agenda,
    semana_de,
    slug,
)

# =============================================================================
# Configuração da página
# =============================================================================
st.set_page_config(
    page_title="Agenda Semanal – CEAD",
    page_icon="📅",
    layout="centered",
)

# =============================================================================
# Cabeçalho visual
# =============================================================================
st.markdown(
    """
    <div style="background:#0B2A3A;padding:24px 28px 18px 28px;border-radius:8px;margin-bottom:24px">
        <h2 style="color:#FFFFFF;margin:0;font-size:1.5rem">
            📅 Gerador de Agenda Semanal
        </h2>
        <p style="color:#AACCDD;margin:6px 0 0 0;font-size:0.95rem">
            Licenciatura em Física – CEAD
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Instruções para o usuário (expansível)
# =============================================================================
with st.expander("ℹ️  Como usar — clique para ver"):
    st.markdown("""
    **Passo 1 — Prepare a planilha Excel** com as seguintes colunas:

    | Coluna | Obrigatório | Exemplo |
    |---|---|---|
    | DATA | ✅ | 26/02/2026 |
    | HORA | ✅ | 19h |
    | ATIVIDADE | ✅ | AULA REMOTA CÁLCULO II |
    | PROFESSOR | — | Prof. Adams |
    | POLO_ATIVIDADE | — | Pedro II / Todos os Polos |
    | POLO | — | Teresina *(para filtrar)* |
    | MODULO | — | III *(para filtrar)* |

    **Passo 2 —** Faça o upload da planilha aqui na página.

    **Passo 3 —** Escolha a semana, o formato e os filtros opcionais.

    **Passo 4 —** Clique em **Gerar Agenda** e baixe as imagens.

    > Se a semana tiver mais de 20 atividades, serão gerados múltiplos
    > cards automaticamente (carrossel), entregues num arquivo ZIP.
    """)

# =============================================================================
# 1) Upload da planilha
# =============================================================================
st.subheader("1 · Planilha de atividades")

arquivo = st.file_uploader(
    "Faça o upload do arquivo Excel (.xlsx)",
    type=["xlsx"],
    help="A planilha deve conter pelo menos as colunas DATA, HORA e ATIVIDADE.",
)

df = None

if arquivo is not None:
    try:
        # Salva temporariamente para usar a função carregar_excel existente
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(arquivo.read())
            tmp_path = tmp.name

        df = carregar_excel(tmp_path)

        # Feedback visual com prévia da planilha
        st.success(f"✅  Planilha carregada — {len(df)} linha(s) encontrada(s).")
        with st.expander("👁️  Prévia da planilha"):
            st.dataframe(df.drop(columns=["DATA_DT"], errors="ignore").head(10),
                         use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao ler a planilha: {e}")

# =============================================================================
# 2) Configurações de geração
# =============================================================================
st.subheader("2 · Configurações")

col1, col2 = st.columns(2)

with col1:
    data_ref = st.date_input(
        "Qualquer data da semana desejada",
        value=date.today(),
        help="O gerador encontra automaticamente a semana (segunda a domingo) que contém esta data.",
    )

with col2:
    formato = st.selectbox(
        "Formato da imagem",
        options=["4:5  —  Feed (Instagram / Facebook)", "9:16  —  Stories / Reels"],
        index=0,
        help="4:5 é o formato padrão de feed. 9:16 é o formato vertical de Stories.",
    )
    aspect = "4:5" if formato.startswith("4:5") else "9:16"

# Filtros opcionais — só aparecem se a planilha tiver as colunas
filtro_polo   = ""
filtro_modulo = ""

if df is not None:
    tem_polo   = "POLO"   in df.columns and df["POLO"].str.strip().ne("").any()
    tem_modulo = "MODULO" in df.columns and df["MODULO"].str.strip().ne("").any()

    if tem_polo or tem_modulo:
        st.markdown("**Filtros opcionais**")
        col3, col4 = st.columns(2)

        if tem_polo:
            polos = ["Todos"] + sorted(df["POLO"].dropna().unique().tolist())
            with col3:
                sel_polo = st.selectbox("Polo", options=polos)
                filtro_polo = "" if sel_polo == "Todos" else sel_polo

        if tem_modulo:
            modulos = ["Todos"] + sorted(df["MODULO"].dropna().unique().tolist())
            with col4:
                sel_modulo = st.selectbox("Módulo", options=modulos)
                filtro_modulo = "" if sel_modulo == "Todos" else sel_modulo

# =============================================================================
# 3) Botão de geração
# =============================================================================
st.subheader("3 · Gerar")

gerar = st.button(
    "🖼️  Gerar Agenda",
    type="primary",
    disabled=(df is None),
    use_container_width=True,
)

if gerar and df is not None:
    try:
        data_ref_ts = pd.Timestamp(data_ref)
        segunda, domingo = semana_de(data_ref_ts)
        label_semana = f"Semana {segunda.strftime('%d/%m')} – {domingo.strftime('%d/%m/%Y')}"

        # Filtra o DataFrame para a semana selecionada
        rec = filtrar_semana(
            df,
            data_ref_ts,
            filtro_polo   or None,
            filtro_modulo or None,
        )

        if rec.empty:
            st.warning("⚠️  Nenhuma atividade encontrada para esta semana com os filtros selecionados.")
            st.stop()

        # Gera os cards em uma pasta temporária
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            base   = f"agenda_{slug(filtro_polo)}_{slug(filtro_modulo)}_{aspect.replace(':','x')}"

            with st.spinner("Gerando cards..."):
                desenhar_card_agenda(
                    rec,
                    polo=filtro_polo,
                    modulo=filtro_modulo,
                    label_semana=label_semana,
                    aspect=aspect,
                    out_png=outdir / f"{base}.png",
                    out_pdf=outdir / f"{base}.pdf",
                )

            # Coleta todos os PNGs gerados (pode ser 1 ou vários se houver carrossel)
            pngs = sorted(outdir.glob("*.png"))
            pdfs = sorted(outdir.glob("*.pdf"))

            if not pngs:
                st.error("Nenhuma imagem foi gerada. Verifique a planilha.")
                st.stop()

            # ── Exibe preview dos cards na página ──
            st.success(f"✅  {len(pngs)} card(s) gerado(s)!")

            for png in pngs:
                st.image(str(png), use_container_width=True)

            # ── Download ──
            st.markdown("---")

            if len(pngs) == 1:
                # Card único — botões separados PNG e PDF
                col_png, col_pdf = st.columns(2)
                with col_png:
                    st.download_button(
                        label="⬇️  Baixar PNG",
                        data=pngs[0].read_bytes(),
                        file_name=pngs[0].name,
                        mime="image/png",
                        use_container_width=True,
                    )
                with col_pdf:
                    st.download_button(
                        label="⬇️  Baixar PDF",
                        data=pdfs[0].read_bytes(),
                        file_name=pdfs[0].name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                # Múltiplos cards — empacota num ZIP com PNGs e PDFs
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in pngs + pdfs:
                        zf.write(f, arcname=f.name)
                zip_buffer.seek(0)

                zip_nome = f"agenda_{slug(filtro_polo)}_{slug(filtro_modulo)}_carrossel.zip"
                st.download_button(
                    label=f"⬇️  Baixar todos os cards (ZIP  –  {len(pngs)} imagens)",
                    data=zip_buffer,
                    file_name=zip_nome,
                    mime="application/zip",
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"Erro durante a geração: {e}")
        st.exception(e)

# =============================================================================
# Rodapé
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#5B6776;font-size:0.85rem'>"
    "Licenciatura em Física · CEAD · cead.ufpi.br"
    "</p>",
    unsafe_allow_html=True,
)

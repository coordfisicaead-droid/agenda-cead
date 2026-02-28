"""
app.py — Interface Streamlit para geração da Agenda Semanal
Licenciatura em Física - CEAD / UFPI
"""

import io
import zipfile
import tempfile
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from gerar_agenda import carregar_excel, paginar, gerar_card, SLOT_H

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agenda Semanal — CEAD/UFPI",
    page_icon="📅",
    layout="centered",
)

# ── Cabeçalho institucional ───────────────────────────────────────────────────
st.markdown("""
<div style="
    background-color:#0B2A3A;
    padding:28px 32px 22px 32px;
    border-radius:8px;
    margin-bottom:28px;
">
    <h2 style="color:#FFFFFF;margin:0;font-size:1.65rem;font-weight:700;">
        📅 Agenda Semanal
    </h2>
    <p style="color:#A8C7DC;margin:6px 0 0 0;font-size:1.0rem;">
        Licenciatura em Física &nbsp;·&nbsp; CEAD / UFPI
    </p>
</div>
""", unsafe_allow_html=True)

st.write("Preencha os campos abaixo e clique em **Gerar cards** para obter as imagens prontas para publicação nas redes sociais.")

st.divider()

# ── Entradas ──────────────────────────────────────────────────────────────────
arquivo = st.file_uploader(
    "📂 Planilha de atividades (.xlsx)",
    type=["xlsx"],
    help="Arquivo Excel com as atividades da semana no formato padrão.",
)

col1, col2 = st.columns(2)

with col1:
    hoje = date.today()
    dias_ate_segunda = (7 - hoje.weekday()) % 7 or 7
    proxima_segunda = hoje + timedelta(days=dias_ate_segunda)

    data_semana = st.date_input(
        "📆 Qualquer data da semana",
        value=proxima_segunda,
        format="DD/MM/YYYY",
        help=(
            "Informe qualquer dia da semana que será divulgada. "
            "O cabeçalho exibirá o intervalo de segunda a domingo dessa semana."
        ),
    )

with col2:
    formato = st.radio(
        "🖼️ Formato do card",
        options=["4:5  —  Feed (padrão)", "9:16  —  Stories / Reels"],
        index=0,
        help="4:5 é o formato padrão para posts no feed do Instagram/Facebook.",
    )

aspect = "9:16" if "9:16" in formato else "4:5"

st.divider()

# ── Botão principal ───────────────────────────────────────────────────────────
gerar = st.button(
    "⚙️  Gerar cards",
    type="primary",
    use_container_width=True,
    disabled=(arquivo is None),
)

if arquivo is None:
    st.caption("⬆️  Faça o upload da planilha para habilitar a geração.")

# ── Processamento ─────────────────────────────────────────────────────────────
if gerar and arquivo:
    with st.spinner("Gerando os cards… aguarde."):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)

                xlsx_path = tmp / "agenda.xlsx"
                xlsx_path.write_bytes(arquivo.read())

                df = carregar_excel(str(xlsx_path))

                if df.empty:
                    st.error("❌ Nenhuma atividade encontrada na planilha. Verifique o arquivo.")
                    st.stop()

                # Intervalo da semana: segunda a domingo da data informada
                seg = data_semana - timedelta(days=data_semana.weekday())
                dom = seg + timedelta(days=6)
                semana_str = f"{seg.strftime('%d/%m')} – {dom.strftime('%d/%m/%Y')}"

                pages = paginar(df, SLOT_H, 1.0)
                total = len(pages)

                cards_bytes = []
                for i, pag_obj in enumerate(pages, start=1):
                    out_png = tmp / f"agenda_p{i:02d}.png"
                    gerar_card(
                        pagina_obj    = pag_obj,
                        semana_str    = semana_str,
                        pagina        = i,
                        total_paginas = total,
                        aspect        = aspect,
                        out_png       = out_png,
                    )
                    cards_bytes.append(out_png.read_bytes())

            # ── Downloads e prévia ────────────────────────────────────────────
            st.success(f"✅ {total} card(s) gerado(s) com sucesso!")

            asp_slug = aspect.replace(":", "x")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, dados in enumerate(cards_bytes, start=1):
                    zf.writestr(f"agenda_semana_p{i:02d}_{asp_slug}.png", dados)
            zip_buf.seek(0)

            st.download_button(
                label="⬇️  Baixar todos os cards (.zip)",
                data=zip_buf,
                file_name=f"agenda_{seg.strftime('%d%m%Y')}_{asp_slug}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

            st.divider()
            st.markdown("### Prévia")

            for i, dados in enumerate(cards_bytes, start=1):
                if total > 1:
                    st.markdown(f"**Card {i} de {total}**")
                st.image(dados, use_container_width=True)
                st.download_button(
                    label=f"⬇️  Baixar card {i}",
                    data=dados,
                    file_name=f"agenda_semana_p{i:02d}_{asp_slug}.png",
                    mime="image/png",
                    key=f"dl_card_{i}",
                )
                if i < total:
                    st.divider()

        except Exception as e:
            st.error(f"❌ Erro ao processar: {e}")
            with st.expander("Detalhes do erro"):
                st.exception(e)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:48px;border-color:#D7DDE5;">
<p style="text-align:center;color:#9AA5B1;font-size:0.80rem;margin:0;">
    CEAD · Centro de Educação Aberta e a Distância · UFPI
</p>
""", unsafe_allow_html=True)

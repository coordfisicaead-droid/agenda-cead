"""
app.py — Interface Streamlit para o gerar_agenda.py
Compatível com share.streamlit.io
"""

import io
import zipfile
import tempfile
from pathlib import Path

import streamlit as st

# ── Importa o script principal ────────────────────────────────────────────────
from gerar_agenda import (
    carregar_excel,
    obter_intervalo_semana,
    paginar,
    gerar_card,
    SLOT_H,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuração da página
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Gerador de Agenda Semanal",
    page_icon="📅",
    layout="centered",
)

st.title("📅 Gerador de Agenda Semanal")
st.caption("CEAD – Centro de Educação Aberta e a Distância | UFPI")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Painel lateral — configurações
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configurações")

    aspect = st.radio(
        "Formato do card",
        options=["4:5", "9:16"],
        index=0,
        help="4:5 para feed do Instagram · 9:16 para Stories/Reels",
    )

    linhas_por_card = st.number_input(
        "Atividades por card (0 = automático)",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Deixe em 0 para usar o cálculo automático baseado em SLOT_H.",
    )

    st.markdown("---")
    st.markdown("**Como usar**")
    st.markdown(
        "1. Faça upload do arquivo Excel\n"
        "2. Ajuste as configurações ao lado\n"
        "3. Clique em **Gerar cards**\n"
        "4. Baixe os PNGs individualmente ou em ZIP"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Upload do arquivo Excel
# ══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "📂 Selecione o arquivo Excel da agenda",
    type=["xlsx"],
    help="Estrutura: col1=ignorada · col2=DATA · col3=HORA · col4+5=DESCRIÇÃO · col6=POLO · col7=PROFESSOR · col8=MÓDULO (opcional)",
)

if uploaded is None:
    st.info("Aguardando upload do arquivo Excel para continuar.")
    st.stop()

# ── Leitura do Excel ──────────────────────────────────────────────────────────
with st.spinner("Lendo o arquivo Excel..."):
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        df = carregar_excel(tmp_path)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {e}")
        st.stop()

if df.empty:
    st.error("Nenhuma atividade encontrada no arquivo. Verifique a estrutura das colunas.")
    st.stop()

# ── Resumo dos dados carregados ───────────────────────────────────────────────
semana_str = obter_intervalo_semana(df)
modulos    = sorted([m for m in df["MODULO"].unique() if m])

col1, col2, col3 = st.columns(3)
col1.metric("Atividades", len(df))
col2.metric("Semana", semana_str or "—")
col3.metric("Módulos", len(modulos) if modulos else "Sem módulo")

if modulos:
    st.info(f"Módulos detectados: **{', '.join(modulos)}**  — será gerada uma agenda por módulo.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Botão de geração
# ══════════════════════════════════════════════════════════════════════════════
if st.button("🚀 Gerar cards", type="primary", use_container_width=True):

    slot_h_ef = 1.0 / linhas_por_card if linhas_por_card > 0 else SLOT_H

    # Monta grupos por módulo (igual ao main() do script original)
    if modulos:
        import re
        grupos = [(m, df[df["MODULO"] == m].reset_index(drop=True))
                  for m in modulos]
    else:
        grupos = [("", df)]

    cards_gerados = []   # lista de (nome_arquivo, bytes_png)

    progress = st.progress(0, text="Gerando cards...")

    # Conta total de cards a gerar
    total_cards = sum(
        len(paginar(df_mod, slot_h_ef, 1.0)) for _, df_mod in grupos
    )
    card_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        for modulo, df_mod in grupos:
            pages    = paginar(df_mod, slot_h_ef, 1.0)
            total    = len(pages)
            mod_slug = re.sub(r"[^A-Za-z0-9]", "_", modulo).lower() if modulo else ""
            prefix   = f"{mod_slug}_" if mod_slug else "agenda_semana_"
            asp_slug = aspect.replace(":", "x")

            for i, pag_obj in enumerate(pages, start=1):
                out_png = outdir / f"{prefix}p{i:02d}_{asp_slug}.png"

                try:
                    gerar_card(
                        pagina_obj    = pag_obj,
                        semana_str    = semana_str,
                        pagina        = i,
                        total_paginas = total,
                        aspect        = aspect,
                        out_png       = out_png,
                        modulo        = modulo,
                    )
                    cards_gerados.append((out_png.name, out_png.read_bytes()))
                except Exception as e:
                    st.warning(f"Erro ao gerar {out_png.name}: {e}")

                card_count += 1
                progress.progress(card_count / total_cards,
                                  text=f"Gerando card {card_count} de {total_cards}…")

    progress.empty()

    if not cards_gerados:
        st.error("Nenhum card foi gerado. Verifique o arquivo Excel.")
        st.stop()

    st.success(f"✅ {len(cards_gerados)} card(s) gerado(s) com sucesso!")
    st.markdown("---")

    # ── Download individual ───────────────────────────────────────────────────
    st.subheader("📥 Download dos cards")

    for nome, dados in cards_gerados:
        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.image(dados, caption=nome, use_container_width=True)
        with col_btn:
            st.download_button(
                label="⬇️ Baixar",
                data=dados,
                file_name=nome,
                mime="image/png",
                key=f"dl_{nome}",
            )

    # ── Download em ZIP ───────────────────────────────────────────────────────
    if len(cards_gerados) > 1:
        st.markdown("---")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for nome, dados in cards_gerados:
                zf.writestr(nome, dados)
        zip_buf.seek(0)

        st.download_button(
            label="📦 Baixar todos em ZIP",
            data=zip_buf,
            file_name=f"agenda_{semana_str.replace('/', '-').replace(' ', '').replace('–','-')}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )

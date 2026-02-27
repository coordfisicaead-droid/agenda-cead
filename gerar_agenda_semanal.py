"""
gerar_agenda_semanal.py
=======================
Gera imagem (PNG) e PDF de Agenda Semanal para divulgação nas redes sociais,
mantendo a identidade visual do gerar_grade.py (cores, cabeçalho, logo).

Planilha Excel esperada
-----------------------
Colunas obrigatórias:
    DATA            – data do evento (ex.: 26/02/2026)
    HORA            – horário (ex.: 19h  ou  19:00)
    ATIVIDADE       – descrição da atividade (ex.: AULA REMOTA CÁLCULO II)

Colunas opcionais:
    PROFESSOR       – nome do professor (ex.: Prof. Adams)
    POLO_ATIVIDADE  – polo(s) destinatário(s) daquela atividade específica
                      (ex.: "Pedro II", "Redenção", "Todos os polos")
                      Quando vazio, a coluna de polo não é exibida na linha.
    POLO            – polo da turma; usado para filtrar com --polo na CLI
    MODULO          – módulo da turma; usado para filtrar com --modulo na CLI

Uso
---
    # Gera agenda para a semana que contém 26/02/2026:
    python gerar_agenda_semanal.py --xlsx agenda.xlsx --semana 26/02/2026

    # Filtra por polo/módulo:
    python gerar_agenda_semanal.py --xlsx agenda.xlsx --semana 26/02/2026 --polo Teresina --modulo III

    # Escolhe o formato Stories (9:16):
    python gerar_agenda_semanal.py --xlsx agenda.xlsx --semana 26/02/2026 --aspect 9:16

    # Gera uma semana para cada combinação polo+módulo presente na planilha:
    python gerar_agenda_semanal.py --xlsx agenda.xlsx --semana 26/02/2026 --lote
"""

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# =============================================================================
# 0)  CONFIGURAÇÃO DA LOGO INSTITUCIONAL  (igual ao gerar_grade.py)
# =============================================================================
LOGO_PATH   = "logo_header_institucional.png"
LOGO_ZOOM   = 1.2
LOGO_POS_X  = 0.85
LOGO_POS_Y  = 0.60

# =============================================================================
# 1)  PALETA  (idêntica ao gerar_grade.py)
# =============================================================================
PALETA = {
    "bg":               "#F7F8FA",
    "header_bg":        "#0B2A3A",
    "header_text":      "#FFFFFF",

    "block_title_bg":   "#123B52",
    "block_title_text": "#FFFFFF",

    "table_header_bg":  "#E9EEF5",
    "table_header_text":"#0E1726",

    "cell_bg":          "#FFFFFF",
    "cell_text":        "#0E1726",

    "grid":             "#D7DDE5",

    "prova_bg":         "#FFE7D6",
    "prova_text":       "#7A2E00",

    "finais_bg":        "#F6DCE3",
    "finais_text":      "#6A1031",

    "muted":            "#5B6776",

    # Cores extras para a agenda
    "dia_bg":           "#1A4F6E",   # faixa do nome do dia (azul médio)
    "dia_text":         "#FFFFFF",
    "hora_text":        "#123B52",   # horário em destaque
    "ativ_text":        "#0E1726",
    "polo_ativ_bg":     "#DDE8F0",   # pílula/badge do polo da atividade
    "polo_ativ_text":   "#0B2A3A",
    "prof_text":        "#5B6776",
    "linha_sep":        "#D7DDE5",
}

# Dias da semana em português (para ordenação e exibição)
DIAS_PT = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}

# =============================================================================
# 2)  UTILITÁRIOS
# =============================================================================
def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s


def semana_de(data_ref: pd.Timestamp):
    """Retorna (segunda, domingo) da semana que contém data_ref."""
    segunda = data_ref - pd.Timedelta(days=data_ref.weekday())
    domingo = segunda + pd.Timedelta(days=6)
    return segunda, domingo


def carregar_excel(path_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(path_xlsx)

    for col in ["DATA", "HORA", "ATIVIDADE", "PROFESSOR", "POLO", "MODULO"]:
        if col in df.columns:
            df[col] = df[col].apply(norm)

    df["DATA_DT"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    return df


# =============================================================================
# 3)  LOGO NO CABEÇALHO  (mesma lógica do gerar_grade.py)
# =============================================================================
def _compor_alpha_sobre_fundo(img_rgba: np.ndarray, hex_bg: str) -> np.ndarray:
    hex_bg = hex_bg.lstrip("#")
    bg = np.array([int(hex_bg[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
    rgb   = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3:4]
    return rgb * alpha + bg * (1 - alpha)


def _inserir_logo_no_header(ax_header) -> bool:
    logo_file = Path(LOGO_PATH)
    if not logo_file.exists():
        return False
    try:
        img = mpimg.imread(str(logo_file))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            alpha_ch = np.ones((*img.shape[:2], 1), dtype=np.float32)
            img = np.concatenate([img, alpha_ch], axis=2)

        img_composta = _compor_alpha_sobre_fundo(img, PALETA["header_bg"])

        h_px, w_px = img_composta.shape[:2]
        aspect_ratio = h_px / w_px
        inset_w = LOGO_ZOOM
        inset_h = inset_w * aspect_ratio
        inset_x = LOGO_POS_X - inset_w / 2
        inset_y = LOGO_POS_Y - inset_h / 2

        ax_logo = ax_header.inset_axes(
            [inset_x, inset_y, inset_w, inset_h],
            transform=ax_header.transAxes,
        )
        ax_logo.set_axis_off()
        ax_logo.patch.set_visible(False)
        ax_logo.imshow(img_composta, aspect="equal", interpolation="lanczos")
        return True
    except Exception as exc:
        print(f"Aviso: não foi possível carregar a logo '{LOGO_PATH}': {exc}")
        return False


# =============================================================================
# 4)  DESENHO DA AGENDA
# =============================================================================
def _desenhar_cabecalho(ax, polo: str, modulo: str, label_semana: str):
    """Cabeçalho idêntico ao gerar_grade.py, com título 'Agenda Semanal'."""
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                color=PALETA["header_bg"],
                                transform=ax.transAxes))

    ax.text(0.03, 0.78, "Licenciatura em Física - CEAD",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=22, fontweight="bold", color=PALETA["header_text"])

    ax.text(0.03, 0.52, "Agenda Semanal",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=15, fontweight="bold", color=PALETA["header_text"])

    info_parts = []
    if polo:
        info_parts.append(f"Polo: {polo}")
    if modulo:
        info_parts.append(f"Módulo {modulo}")
    info_parts.append(label_semana)
    ax.text(0.03, 0.22, "  |  ".join(info_parts),
            transform=ax.transAxes, ha="left", va="center",
            fontsize=12, color=PALETA["header_text"])

    _inserir_logo_no_header(ax)


# =============================================================================
# Constantes de layout — coluna lateral (Opção A aprovada)
# =============================================================================
W_DIA    = 0.120   # 13% da largura para coluna do dia
COL_HORA = 0.121   # início da coluna de hora (logo após a coluna do dia)
W_HORA   = 0.065
COL_ATIV = COL_HORA + W_HORA + 0.005
COL_POLO = 0.525
COL_PROF = 0.790


def _calcular_linha_h(grupos: list[tuple]) -> float:
    """Altura de cada linha de atividade, ajustada dinamicamente ao conteúdo."""
    total_linhas = sum(len(ativs) for _, _, ativs in grupos)
    sep_total    = len(grupos) * 0.004   # separadores finos entre dias
    linha_h_raw  = (1.0 - sep_total) / max(total_linhas, 1)
    return max(0.032, min(linha_h_raw, 0.080))


def _desenhar_coluna_dia(ax_body, y_top: float, bloco_h: float,
                          nome_dia: str, data_str: str):
    """Coluna azul lateral com nome do dia e data centralizados no bloco."""
    ax_body.add_patch(mpatches.FancyBboxPatch(
        (0.0, y_top - bloco_h), W_DIA, bloco_h,
        boxstyle="square,pad=0",
        facecolor=PALETA["dia_bg"],
        edgecolor="none",
        transform=ax_body.transAxes,
        clip_on=True,
    ))
    ax_body.text(W_DIA / 2, y_top - bloco_h / 2,
                 f"{nome_dia}\n{data_str}",
                 transform=ax_body.transAxes,
                 ha="center", va="center",
                 fontsize=9.5, fontweight="bold",
                 color=PALETA["dia_text"],
                 linespacing=1.4)


def _desenhar_atividade(ax_body, y_cursor: float, linha_h: float,
                         hora: str, atividade: str, polo_ativ: str,
                         professor: str, idx: int):
    """Renderiza uma linha de atividade: hora | atividade | [polo] | professor."""
    bg = PALETA["cell_bg"] if idx % 2 == 0 else PALETA["table_header_bg"]
    # Fundo da linha (apenas na área à direita da coluna do dia)
    ax_body.add_patch(mpatches.FancyBboxPatch(
        (COL_HORA, y_cursor - linha_h), 1.0 - COL_HORA, linha_h,
        boxstyle="square,pad=0",
        facecolor=bg,
        edgecolor=PALETA["grid"],
        linewidth=0.4,
        transform=ax_body.transAxes,
        clip_on=True,
    ))

    y_mid = y_cursor - linha_h / 2

    # Hora
    ax_body.text(COL_HORA + 0.007, y_mid, hora,
                 transform=ax_body.transAxes,
                 ha="left", va="center",
                 fontsize=11, fontweight="bold",
                 color=PALETA["hora_text"])

    # Atividade
    ax_body.text(COL_ATIV, y_mid, atividade,
                 transform=ax_body.transAxes,
                 ha="left", va="center",
                 fontsize=10,
                 color=PALETA["ativ_text"])

    # Polo da atividade — badge (só se preenchido)
    if polo_ativ:
        ax_body.text(COL_POLO + 0.075, y_mid, polo_ativ,
                     transform=ax_body.transAxes,
                     ha="center", va="center",
                     fontsize=8.5, fontweight="bold",
                     color=PALETA["polo_ativ_text"],
                     bbox=dict(
                         boxstyle="round,pad=0.22",
                         facecolor=PALETA["polo_ativ_bg"],
                         edgecolor=PALETA["grid"],
                         linewidth=0.7,
                     ))

    # Professor (alinhado à direita)
    if professor:
        ax_body.text(0.995, y_mid, f"({professor})",
                     transform=ax_body.transAxes,
                     ha="right", va="center",
                     fontsize=9, style="italic",
                     color=PALETA["prof_text"])


def desenhar_corpo_agenda(ax, grupos: list[tuple]):
    """
    grupos: [(nome_dia, data_str, [(hora, atividade, polo_ativ, professor), ...]), ...]
    Layout Opção A: coluna azul lateral (13%) + linhas de atividade à direita.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    linha_h = _calcular_linha_h(grupos)
    sep_h   = 0.004   # separador fino entre dias

    y = 1.0
    for nome_dia, data_str, atividades in grupos:
        bloco_h = len(atividades) * linha_h

        # Coluna lateral azul do dia
        _desenhar_coluna_dia(ax, y, bloco_h, nome_dia, data_str)

        # Linhas de atividade
        for idx, (hora, atividade, polo_ativ, professor) in enumerate(atividades):
            _desenhar_atividade(ax, y, linha_h, hora, atividade, polo_ativ, professor, idx)
            y -= linha_h

        # Separador fino entre blocos de dias
        ax.plot([0, 1], [y, y], color=PALETA["grid"], linewidth=1.0,
                transform=ax.transAxes, clip_on=True)
        y -= sep_h


def _rodape(ax):
    ax.set_axis_off()
    ax.text(0.03, 0.65,
            "Acompanhe nossas redes sociais para mais informações.",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=9, color=PALETA["muted"])
    ax.text(0.03, 0.25,
            "cead.ufpi.br",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=9, fontweight="bold", color=PALETA["muted"])


# =============================================================================
# 5)  MONTAGEM DO CARD
# =============================================================================
def montar_grupos(df_semana: pd.DataFrame) -> list[tuple]:
    """
    Agrupa as atividades da semana por dia.
    Retorna lista ordenada de (nome_dia, data_str, [(hora, atividade, polo_ativ, professor), ...]).
    """
    has_polo_ativ = "POLO_ATIVIDADE" in df_semana.columns

    grupos = []
    dias_presentes = sorted(df_semana["DATA_DT"].dropna().unique())

    for data_np in dias_presentes:
        data_ts = pd.Timestamp(data_np)
        dow = data_ts.weekday()
        nome_dia  = DIAS_PT.get(dow, "?")
        data_str  = data_ts.strftime("%d/%m/%y")

        dia_df = df_semana[df_semana["DATA_DT"] == data_np].copy()
        sort_cols = ["HORA"]
        if has_polo_ativ:
            sort_cols.append("POLO_ATIVIDADE")
        dia_df = dia_df.sort_values(sort_cols)

        atividades = []
        for _, row in dia_df.iterrows():
            hora       = norm(row["HORA"])
            ativ       = norm(row["ATIVIDADE"]).upper()
            polo_ativ  = norm(row["POLO_ATIVIDADE"]) if has_polo_ativ else ""
            prof       = norm(row.get("PROFESSOR", ""))
            atividades.append((hora, ativ, polo_ativ, prof))

        if atividades:
            grupos.append((nome_dia, data_str, atividades))

    return grupos


# Número máximo de linhas de atividade por card antes de paginar.
# Com linha_h mínima de 0.032 e separadores, ~22 linhas preenchem bem o 4:5.
# Deixamos uma margem segura em 20 para garantir conforto visual.
MAX_LINHAS_POR_CARD = 20


def _paginar_grupos(grupos: list[tuple]) -> list[list[tuple]]:
    """
    Divide a lista de grupos em páginas de no máximo MAX_LINHAS_POR_CARD linhas.

    Regra: um dia nunca é partido entre duas páginas — ele vai inteiro para a
    próxima página se não couber na atual. Isso preserva a coerência visual.
    """
    paginas: list[list[tuple]] = []
    pagina_atual: list[tuple]  = []
    linhas_atual = 0

    for nome_dia, data_str, atividades in grupos:
        n = len(atividades)

        # Se o dia sozinho já passa do limite, coloca numa página exclusiva
        if n > MAX_LINHAS_POR_CARD:
            if pagina_atual:
                paginas.append(pagina_atual)
                pagina_atual = []
                linhas_atual = 0
            paginas.append([(nome_dia, data_str, atividades)])
            continue

        # Se não cabe na página atual, abre nova página
        if linhas_atual + n > MAX_LINHAS_POR_CARD:
            paginas.append(pagina_atual)
            pagina_atual = []
            linhas_atual = 0

        pagina_atual.append((nome_dia, data_str, atividades))
        linhas_atual += n

    if pagina_atual:
        paginas.append(pagina_atual)

    return paginas


def _renderizar_card(grupos_pagina: list[tuple], polo: str, modulo: str,
                      label_semana: str, aspect: str,
                      out_png: Path, out_pdf: Path,
                      pagina: int = 0, total_paginas: int = 1):
    """Renderiza um único card (uma página do carrossel)."""
    if aspect == "9:16":
        fig_w, fig_h = 10.8, 19.2
    else:
        fig_w, fig_h = 10.8, 13.5   # 4:5

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor(PALETA["bg"])

    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[0.14, 0.80, 0.06],
        hspace=0.06,
    )

    # Cabeçalho — acrescenta indicador de página quando há mais de um card
    ax_h = fig.add_subplot(gs[0])
    if total_paginas > 1:
        label_pag = f"{label_semana}   •   {pagina}/{total_paginas}"
    else:
        label_pag = label_semana
    _desenhar_cabecalho(ax_h, polo, modulo, label_pag)

    # Corpo
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor(PALETA["bg"])
    desenhar_corpo_agenda(ax_b, grupos_pagina)

    # Rodapé
    ax_r = fig.add_subplot(gs[2])
    ax_r.set_facecolor(PALETA["bg"])
    _rodape(ax_r)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Card {pagina}/{total_paginas} → {out_png.name}")


def desenhar_card_agenda(
    df_semana: pd.DataFrame,
    polo: str,
    modulo: str,
    label_semana: str,
    aspect: str,
    out_png: Path,
    out_pdf: Path,
):
    """
    Gera um ou mais cards (carrossel automático) conforme o volume de atividades.
    Se o total de linhas couber em um único card, gera apenas um arquivo.
    Se não couber, gera  base_p1.png, base_p2.png … preservando dias inteiros
    em cada página.
    """
    grupos = montar_grupos(df_semana)

    if not grupos:
        # Nenhuma atividade — gera card vazio informativo
        fig_w, fig_h = (10.8, 19.2) if aspect == "9:16" else (10.8, 13.5)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        fig.patch.set_facecolor(PALETA["bg"])
        gs  = fig.add_gridspec(nrows=3, ncols=1,
                               height_ratios=[0.14, 0.80, 0.06], hspace=0.06)
        _desenhar_cabecalho(fig.add_subplot(gs[0]), polo, modulo, label_semana)
        ax_b = fig.add_subplot(gs[1])
        ax_b.set_axis_off()
        ax_b.text(0.5, 0.5, "Nenhuma atividade nesta semana.",
                  transform=ax_b.transAxes, ha="center", va="center",
                  fontsize=14, color=PALETA["muted"])
        _rodape(fig.add_subplot(gs[2]))
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"OK (sem atividades): {out_png.name}")
        return

    paginas = _paginar_grupos(grupos)
    total   = len(paginas)

    # Base do nome de arquivo sem extensão
    base_sem_ext = out_png.with_suffix("")

    for i, grupos_pag in enumerate(paginas, start=1):
        if total == 1:
            # Mantém o nome original (sem sufixo _p1) para não quebrar uso já existente
            png_i = out_png
            pdf_i = out_pdf
        else:
            png_i = base_sem_ext.parent / f"{base_sem_ext.name}_p{i}.png"
            pdf_i = out_pdf.with_suffix("").parent / f"{out_pdf.with_suffix('').name}_p{i}.pdf"

        _renderizar_card(grupos_pag, polo, modulo, label_semana,
                         aspect, png_i, pdf_i,
                         pagina=i, total_paginas=total)

    if total > 1:
        print(f"  → Carrossel gerado: {total} cards para '{out_png.stem}'.")


# =============================================================================
# 6)  FILTRO DE SEMANA
# =============================================================================
def filtrar_semana(df: pd.DataFrame, data_ref: pd.Timestamp,
                   polo: str | None, modulo: str | None) -> pd.DataFrame:
    segunda, domingo = semana_de(data_ref)
    mask = (df["DATA_DT"] >= segunda) & (df["DATA_DT"] <= domingo)
    if polo and "POLO" in df.columns:
        mask &= df["POLO"].str.upper() == polo.upper()
    if modulo and "MODULO" in df.columns:
        mask &= df["MODULO"].str.upper() == modulo.upper()
    return df[mask].copy()


# =============================================================================
# 7)  MODO LOTE
# =============================================================================
def gerar_lote(df: pd.DataFrame, data_ref: pd.Timestamp,
               outdir: Path, aspect: str):
    segunda, domingo = semana_de(data_ref)
    label_semana = f"Semana {segunda.strftime('%d/%m')} – {domingo.strftime('%d/%m/%Y')}"

    has_polo   = "POLO"   in df.columns
    has_modulo = "MODULO" in df.columns

    if has_polo and has_modulo:
        combos = (df[["POLO", "MODULO"]].drop_duplicates()
                    .sort_values(["MODULO", "POLO"])
                    .to_dict("records"))
    elif has_polo:
        combos = [{"POLO": p, "MODULO": ""} for p in df["POLO"].unique()]
    elif has_modulo:
        combos = [{"POLO": "", "MODULO": m} for m in df["MODULO"].unique()]
    else:
        combos = [{"POLO": "", "MODULO": ""}]

    print(f"Modo lote: {len(combos)} turma(s) encontradas.")
    for c in combos:
        polo, modulo = c.get("POLO", ""), c.get("MODULO", "")
        rec = filtrar_semana(df, data_ref, polo or None, modulo or None)
        if rec.empty:
            print(f"PULADO (sem dados): {polo} | {modulo}")
            continue

        base = f"agenda_{slug(polo)}_{slug(modulo)}_{aspect.replace(':','x')}"
        desenhar_card_agenda(rec, polo, modulo, label_semana, aspect,
                             outdir / f"{base}.png", outdir / f"{base}.pdf")


# =============================================================================
# 8)  CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Gera Agenda Semanal (PNG/PDF) para redes sociais."
    )
    ap.add_argument("--xlsx",    default="agenda_semanal.xlsx",
                    help="Caminho do arquivo .xlsx  (padrão: agenda_semanal.xlsx)")
    ap.add_argument("--semana",  required=True,
                    help="Qualquer data da semana desejada, formato DD/MM/AAAA  (ex.: 26/02/2026)")
    ap.add_argument("--polo",    default=None,
                    help="Filtro de polo  (opcional)")
    ap.add_argument("--modulo",  default=None,
                    help="Filtro de módulo  (opcional)")
    ap.add_argument("--lote",    action="store_true",
                    help="Gera um card para cada combinação polo+módulo da planilha")
    ap.add_argument("--aspect",  choices=["4:5", "9:16"], default="4:5",
                    help="Formato da arte: 4:5 (feed) ou 9:16 (stories)  (padrão: 4:5)")
    ap.add_argument("--outdir",  default="saida",
                    help="Pasta de saída  (padrão: saida/)")
    args = ap.parse_args()

    # Valida a data de referência
    try:
        data_ref = pd.to_datetime(args.semana, dayfirst=True)
    except Exception:
        raise SystemExit(f"Data inválida: '{args.semana}'. Use DD/MM/AAAA.")

    df = carregar_excel(args.xlsx)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    segunda, domingo = semana_de(data_ref)
    label_semana = f"Semana {segunda.strftime('%d/%m')} – {domingo.strftime('%d/%m/%Y')}"

    if args.lote:
        gerar_lote(df, data_ref, outdir, args.aspect)
        return

    rec = filtrar_semana(df, data_ref, args.polo, args.modulo)
    if rec.empty:
        raise SystemExit("Nenhum dado encontrado para essa semana/filtro. Verifique a planilha.")

    polo   = args.polo   or ""
    modulo = args.modulo or ""
    base   = f"agenda_{slug(polo)}_{slug(modulo)}_{args.aspect.replace(':','x')}"

    desenhar_card_agenda(
        rec, polo, modulo, label_semana, args.aspect,
        outdir / f"{base}.png",
        outdir / f"{base}.pdf",
    )


if __name__ == "__main__":
    main()

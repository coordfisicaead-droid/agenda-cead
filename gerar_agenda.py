"""
gerar_agenda.py
---------------
Gera cards PNG de Agenda Semanal para redes sociais,
com o mesmo padrão visual do gerar_grade.py (paleta e logo institucionais).

Uso:
    python gerar_agenda.py --xlsx agenda.xlsx
    python gerar_agenda.py --xlsx agenda.xlsx --aspect 9:16 --outdir saida
    python gerar_agenda.py --xlsx agenda.xlsx --linhas-por-card 10

Estrutura do Excel (colunas por posição, a 1ª coluna é ignorada):
    col 0 → ignorada
    col 1 → DATA DA ATIVIDADE      (dd/mm/yyyy ou date)
    col 2 → HORA DA ATIVIDADE      (ex.: "08:00" ou "08:00 – 12:00")
    col 3 → DESCRIÇÃO parte 1      ⎫ concatenadas com espaço
    col 4 → DESCRIÇÃO parte 2      ⎭  → DESCRIÇÃO DA ATIVIDADE
    col 5 → POLO DA ATIVIDADE      (ex.: "Pedro II", "Todos os Polos"; vazio = sem badge)
    col 6 → NOME DO PROFESSOR
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

# ══════════════════════════════════════════════════════════════════════════════
# 0) CONFIGURAÇÃO INSTITUCIONAL  (mesma interface do gerar_grade.py)
# ══════════════════════════════════════════════════════════════════════════════
LOGO_PATH   = "logo_header_institucional.png"   # None desativa
LOGO_ZOOM   = 1.2
LOGO_POS_X  = 0.85
LOGO_POS_Y  = 0.60

# Altura de cada linha de atividade, como fração da área do corpo do card.
# Este é o ÚNICO valor que você precisa ajustar — paginação e altura são derivadas dele.
# Exemplos:
#   0.09 -> linhas generosas  (~8 atividades por card no formato 4:5)
#   0.10 -> padrão equilibrado (~7 atividades por card)
#   0.12 -> linhas mais altas  (~6 atividades por card)
SLOT_H = 0.11   # <- AJUSTE AQUI  (slots generosos como na referência)

# Proporções verticais do card (devem somar 1.0)
H_HEADER    = 0.14
H_COLHEADER = 0.04
H_FOOTER    = 0.07
H_CORPO     = 1.0 - H_HEADER - H_COLHEADER - H_FOOTER  # ~0.75

# O axes do corpo usa coordenadas 0-1 internamente (padrão matplotlib).
# SLOT_H é uma fração dessas coordenadas → cabem int(1.0 / SLOT_H) slots.
# A paginação usa exatamente 1.0 como altura disponível — sem conversões.
LINHAS_ATIVIDADE_POR_CARD = max(1, int(1.0 / SLOT_H))

# ══════════════════════════════════════════════════════════════════════════════
# 1) PALETA  (idêntica ao gerar_grade.py)
# ══════════════════════════════════════════════════════════════════════════════
PALETA = {
    # Fundo geral — branco puro
    "bg":               "#FFFFFF",

    # Cabeçalho — azul escuro como no original e na referência
    "header_bg":        "#0B2A3A",
    "header_text":      "#FFFFFF",

    "block_title_bg":   "#123B52",
    "block_title_text": "#FFFFFF",

    # Header de colunas — cinza claro, texto escuro
    "table_header_bg":  "#EBEBEB",
    "table_header_text":"#1A1A2E",

    # Células — branco puro, sem alternância
    "cell_bg":          "#FFFFFF",
    "cell_alt_bg":      "#FFFFFF",
    "cell_text":        "#1A1A2E",

    # Grade — cinza suave
    "grid":             "#CCCCCC",

    # Coluna lateral do dia — azul escuro como na referência
    "day_col_bg":       "#0B2A3A",
    "day_col_text":     "#FFFFFF",

    # Badge de polo — fundo branco, borda cinza (estilo referência)
    "badge_bg":         "#EBEBEB",  ## JRM TROCOU #FFFFFF 
    "badge_text":       "#1A1A2E",
    "badge_border":     "#999999",

    # Horário — vermelho coral como na referência
    "hour_text":        "#E8352A",

    # Professor — cinza médio itálico
    "prof_text":        "#666666",
    "muted":            "#666666",

    # Badge de módulo — vermelho coral, texto branco
    "modulo_bg":        "#E8352A",
    "modulo_text":      "#FFFFFF",
    "modulo_border":    "#C42B20",
}

DIAS_PT = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}

# ══════════════════════════════════════════════════════════════════════════════
# 2) UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════
def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _compor_alpha(img: np.ndarray, cor_fundo_hex: str) -> np.ndarray:
    rgb   = img[:, :, :3]
    alpha = img[:, :, 3:4]
    fundo = _hex_to_rgb(cor_fundo_hex)
    return np.clip(alpha * rgb + (1 - alpha) * fundo, 0, 1).astype(np.float32)


def _inserir_logo(ax_header):
    if not LOGO_PATH:
        return False
    logo_file = Path(LOGO_PATH)
    if not logo_file.exists():
        print(f"Aviso: logo não encontrada: '{LOGO_PATH}'")
        return False
    try:
        img = mpimg.imread(str(logo_file))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.concatenate([img, np.ones((*img.shape[:2], 1), np.float32)], axis=2)
        img = _compor_alpha(img, PALETA["header_bg"])
        h_px, w_px = img.shape[:2]
        inset_w = LOGO_ZOOM
        inset_h = inset_w * (h_px / w_px)
        inset_x = LOGO_POS_X - inset_w / 2
        inset_y = LOGO_POS_Y - inset_h / 2
        ax_logo = ax_header.inset_axes(
            [inset_x, inset_y, inset_w, inset_h],
            transform=ax_header.transAxes,
        )
        ax_logo.set_axis_off()
        ax_logo.patch.set_visible(False)
        ax_logo.imshow(img, aspect="equal", interpolation="lanczos")
        return True
    except Exception as exc:
        print(f"Aviso: erro ao carregar logo: {exc}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# 3) LEITURA DO EXCEL
# ══════════════════════════════════════════════════════════════════════════════
def _parse_data(valor: str):
    """Converte string de data sem gerar UserWarning do pandas."""
    if not valor:
        return pd.NaT
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.to_datetime(valor, format=fmt)
        except (ValueError, TypeError):
            pass
    try:
        return pd.to_datetime(valor, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT


def carregar_excel(path_xlsx: str) -> pd.DataFrame:
    """Lê o Excel ignorando a 1ª coluna e atribui nomes fixos às colunas.

    Estrutura esperada (por posição):
        col 0 -> ignorada | col 1 -> DATA | col 2 -> HORA
        col 3+4 -> DESCRICAO | col 5 -> POLO | col 6 -> PROFESSOR
        col 7 -> MODULO (opcional: "III", "VII", "Módulo III" etc.)
    """
    raw = pd.read_excel(path_xlsx, header=0)

    while len(raw.columns) < 8:
        raw[f"_extra_{len(raw.columns)}"] = np.nan

    df = pd.DataFrame()
    df["DATA"]      = raw.iloc[:, 1].apply(norm)
    df["HORA"]      = raw.iloc[:, 2].apply(norm)
    desc1           = raw.iloc[:, 3].apply(norm)
    desc2           = raw.iloc[:, 4].apply(norm)
    df["DESCRICAO"] = desc1.str.cat(desc2.where(desc2 != "", other=""), sep=" ", na_rep="").str.strip()
    df["POLO"]      = raw.iloc[:, 5].apply(norm)
    df["PROFESSOR"] = raw.iloc[:, 6].apply(norm)

    def _norm_modulo(x) -> str:
        val = norm(x)
        if not val:
            return ""
        val = re.sub(r"(?i)^m[oó]dulo\s*", "", val).strip()
        return f"Módulo {val.upper()}"

    df["MODULO"] = raw.iloc[:, 7].apply(_norm_modulo)

    df["DATA_DT"] = df["DATA"].apply(_parse_data)
    df = df.dropna(subset=["DATA_DT"]).copy()

    df["_hora_sort"] = df["HORA"].str.extract(r"(\d{1,2}:\d{2})")[0]
    df["_hora_sort"] = pd.to_datetime(df["_hora_sort"], format="%H:%M", errors="coerce")
    df = df.sort_values(["DATA_DT", "_hora_sort", "POLO"]).reset_index(drop=True)

    return df


def obter_intervalo_semana(df: pd.DataFrame) -> str:
    """Retorna string 'DD/MM – DD/MM/AAAA' baseada nas datas do DataFrame."""
    datas = df["DATA_DT"].dropna()
    if datas.empty:
        return ""
    d_min = datas.min()
    d_max = datas.max()
    if d_min == d_max:
        return d_min.strftime("%d/%m/%Y")
    return f"{d_min.strftime('%d/%m')} – {d_max.strftime('%d/%m/%Y')}"


# ══════════════════════════════════════════════════════════════════════════════
# 4) PAGINAÇÃO  (altura fixa — dia pode ser quebrado entre cards)
# ══════════════════════════════════════════════════════════════════════════════

# Estrutura de uma fatia de página
# campos: df (atividades), data_dt, is_continuation (vem do card anterior),
#         has_continuation (continua no próximo card)
from dataclasses import dataclass, field as dc_field

@dataclass
class FatiaDia:
    df:               pd.DataFrame
    data_dt:          object
    is_continuation:  bool = False   # este bloco vem do card anterior
    has_continuation: bool = False   # este bloco continua no próximo card

@dataclass
class Pagina:
    fatias: list = dc_field(default_factory=list)
    altura_usada: float = 0.0

    def n_ativ(self):
        return sum(len(f.df) for f in self.fatias)


def paginar(df: pd.DataFrame, slot_h: float, altura_corpo: float) -> list["Pagina"]:
    """
    Distribui as atividades em páginas com SLOT_H fixo.
    Um dia pode ser quebrado entre cards; indicadores visuais marcam
    is_continuation e has_continuation em cada fatia.
    """
    grupos = [(dt, grp.reset_index(drop=True))
              for dt, grp in df.groupby("DATA_DT", sort=True)]

    pages   = [Pagina()]

    for data_dt, grp in grupos:
        restante = list(grp.iterrows())  # lista de (idx, row)
        primeiro_bloco = True

        while restante:
            pag = pages[-1]
            slots_livres = int((altura_corpo - pag.altura_usada + 1e-9) / slot_h)

            if slots_livres <= 0:
                pages.append(Pagina())
                continue

            bloco_rows  = restante[:slots_livres]
            restante    = restante[slots_livres:]

            is_cont  = not primeiro_bloco
            has_cont = len(restante) > 0

            fatia = FatiaDia(
                df               = pd.DataFrame([r for _, r in bloco_rows]),
                data_dt          = data_dt,
                is_continuation  = is_cont,
                has_continuation = has_cont,
            )
            pag.fatias.append(fatia)
            pag.altura_usada += len(bloco_rows) * slot_h
            primeiro_bloco = False

            if has_cont:
                pages.append(Pagina())

    # Remove páginas vazias
    pages = [p for p in pages if p.fatias]
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# 5) RENDERIZAÇÃO DO CARD
# ══════════════════════════════════════════════════════════════════════════════
def _desenhar_cabecalho(ax, semana_str: str, pagina: int, total: int,
                        modulo: str = ""):
    ax.set_axis_off()
    # Fundo azul escuro — igual à referência
    ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                color=PALETA["header_bg"],
                                transform=ax.transAxes))

    # Linha 1 — título (branco, negrito)
    ax.text(0.03, 0.76, "Licenciatura em Física - CEAD",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=20, fontweight="bold", color=PALETA["header_text"])

    # Linha 2 — "Agenda Semanal" (branco, sem negrito, menor)
    ax.text(0.03, 0.50, "Agenda Semanal",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=13, color=PALETA["header_text"])

    if modulo:
        # Badge vermelho coral — largura reduzida 5% (0.27 → 0.256)
        # x ajustado para manter centralização: 0.35 + (0.27-0.256)/2 ≈ 0.357
        ax.add_patch(FancyBboxPatch(
            (0.357, 0.26), 0.256, 0.34,
            boxstyle="round,pad=0.02",
            facecolor=PALETA["modulo_bg"],
            edgecolor=PALETA["modulo_border"],
            linewidth=1.0,
            transform=ax.transAxes,
            zorder=3,
        ))
        # Centro horizontal: 0.357 + 0.256/2 = 0.485 (inalterado)
        ax.text(0.485, 0.43, modulo,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=PALETA["modulo_text"],
                zorder=4)

    # Linha 3 — semana + paginação: descida para 0.12 para não colidir
    # com o badge de módulo quando há paginação ("Página X de Y")
    pag_str = f"  •  Página {pagina} de {total}" if total > 1 else ""
    ax.text(0.03, 0.12, f"Semana {semana_str}{pag_str}",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=11, color=PALETA["header_text"])

    _inserir_logo(ax)


def _desenhar_header_colunas(ax):
    """Linha de cabeçalho das colunas (dia | hora | atividade | polo)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                color=PALETA["table_header_bg"],
                                transform=ax.transAxes))

    # Proporções das colunas ajustadas para a referência visual
    X_DIA  = 0.0;   W_DIA  = 0.16
    X_HORA = 0.16;  W_HORA = 0.13
    X_DESC = 0.29;  W_DESC = 0.48
    X_POLO = 0.77;  W_POLO = 0.23

    labels = [
        ("DIA / DATA",             X_DIA  + W_DIA  / 2),
        ("HORÁRIO",                X_HORA + W_HORA / 2),
        ("DESCRIÇÃO / PROFESSOR",  X_DESC + W_DESC / 2),
        ("POLO",                   X_POLO + W_POLO / 2),
    ]

    for txt, x in labels:
        ax.text(x, 0.5, txt,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=PALETA["table_header_text"])

    # Linha inferior
    ax.axhline(0, color=PALETA["grid"], linewidth=1.2, xmin=0, xmax=1)

    # Linhas verticais separando colunas (mesma cor/espessura das horizontais)
    for x in [X_HORA, X_DESC, X_POLO]:
        ax.axvline(x, color=PALETA["grid"], linewidth=0.8, ymin=0, ymax=1)


def _badge_polo(ax, x_left, y_center, texto, ax_width_pts):
    """Desenha um badge arredondado com o nome do polo."""
    if not texto:
        return
    # Posição em coordenadas de axes
    pad_x = 0.008
    pad_y = 0.12
    fancy = FancyBboxPatch(
        (x_left, y_center - pad_y),
        0.18,          # largura fixa normalizada
        2 * pad_y,
        boxstyle="round,pad=0.01",
        facecolor=PALETA["badge_bg"],
        edgecolor=PALETA["badge_text"],
        linewidth=0.6,
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(fancy)
    ax.text(x_left + 0.09, y_center, texto,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=7.5, color=PALETA["badge_text"],
            fontweight="bold", zorder=4,
            clip_on=True)


def _medir_texto_px(texto: str, fontsize: float, fig_w_in: float,
                    fig_h_in: float, dpi: float, bold: bool = True) -> float:
    """Retorna a largura aproximada do texto em pixels usando a fonte padrão."""
    import matplotlib.font_manager as fm
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    prop = fm.FontProperties(size=fontsize,
                             weight="bold" if bold else "normal")
    path = TextPath((0, 0), texto, prop=prop)
    bb   = path.get_extents()
    # TextPath usa unidades de ponto (72 pt = 1 inch)
    return bb.width / 72.0 * dpi


def _quebrar_polo_em_linhas(polo: str, fontsize: float,
                             max_px: float, fig_w_in: float,
                             fig_h_in: float, dpi: float) -> list[str]:
    """
    Divide o texto do polo em linhas que caibam em max_px.
    Quebra sempre em fronteiras de palavras completas.
    Retorna lista de strings (1 ou 2 linhas).
    """
    palavras = polo.split()
    linhas   = []
    atual    = []

    for palavra in palavras:
        tentativa = " ".join(atual + [palavra])
        w = _medir_texto_px(tentativa, fontsize, fig_w_in, fig_h_in, dpi)
        if atual and w > max_px:
            linhas.append(" ".join(atual))
            atual = [palavra]
        else:
            atual.append(palavra)

    if atual:
        linhas.append(" ".join(atual))

    return linhas


def _renderizar_polo_badge(ax, polo: str, X_POLO: float, W_POLO: float,
                            y_mid: float, slot_h: float,
                            fig_size_inches, dpi: float):
    """
    Renderiza UMA única caixa arredondada (FancyBboxPatch) por slot,
    com todo o texto do polo dentro — quebrado em linhas se necessário.
    A caixa é centralizada verticalmente no slot e ocupa ~80% da largura
    da coluna, com margem lateral simétrica.
    """
    fig_w_in, fig_h_in = fig_size_inches

    # Largura disponível na coluna polo em pixels (para calcular quebra de linha)
    axes_w_px = fig_w_in * dpi * 0.96
    col_px    = W_POLO * axes_w_px * 0.78      # 78% da coluna para o texto

    fonte_base = 11.0
    fonte_min  = 7.5

    # Determina fonte e linhas que cabem
    fonte = fonte_base
    while fonte >= fonte_min:
        linhas = _quebrar_polo_em_linhas(polo, fonte, col_px,
                                          fig_w_in, fig_h_in, dpi)
        todas_cabem = all(
            _medir_texto_px(l, fonte, fig_w_in, fig_h_in, dpi) <= col_px
            for l in linhas
        )
        if todas_cabem and len(linhas) <= 3:
            break
        fonte -= 0.5

    if len(linhas) > 3:
        linhas = [linhas[0], linhas[1], " ".join(linhas[2:])]

    n_linhas   = len(linhas)
    x_centro   = X_POLO + W_POLO / 2

    # ── Dimensões da caixa única ───────────────────────────────────────────
    # Largura: 80% da coluna — borda bem próxima do texto
    box_w  = W_POLO * 0.80
    box_x  = X_POLO + (W_POLO - box_w) / 2

    # Altura: padding mínimo para colar a borda no texto
    line_h = slot_h * 0.20          # altura por linha de texto
    pad_v  = slot_h * 0.03          # padding vertical mínimo
    box_h  = n_linhas * line_h + 2 * pad_v
    box_h  = min(box_h, slot_h * 0.75)   # nunca maior que 75% do slot
    box_y  = y_mid - box_h / 2           # centralizada no slot

    # ── Desenha a caixa única ─────────────────────────────────────────────
    # pad=0.008 no boxstyle controla o raio interno do arredondamento —
    # mantém cantos bem arredondados sem inflar a caixa para fora
    ax.add_patch(FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle="round,pad=0.008",
        facecolor=PALETA["badge_bg"],
        edgecolor=PALETA["badge_border"],
        linewidth=1.1,
        transform=ax.transAxes,
        zorder=3,
        clip_on=True,
    ))

    # ── Texto: uma linha por vez, centralizado dentro da caixa ────────────
    # Distribui as linhas uniformemente dentro da altura da caixa
    if n_linhas == 1:
        y_posicoes = [y_mid]
    else:
        espaco = box_h / (n_linhas + 1)
        y_posicoes = [box_y + box_h - espaco * (k + 1) for k in range(n_linhas)]

    for linha, y_pos in zip(linhas, y_posicoes):
        ax.text(
            x_centro, y_pos,
            linha,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=fonte, fontweight="bold",
            color=PALETA["badge_text"],
            clip_on=True,
            zorder=4,
        )


def _desenhar_corpo(ax, fatias: list, n_slots: int, fig=None):
    """
    Desenha o corpo da agenda a partir de uma lista de FatiaDia.
    slot_h é sempre SLOT_H — fixo, independente do número de atividades.
    fig é necessário para medir texto do badge polo com precisão.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    slot_h = SLOT_H

    X_DIA   = 0.0
    W_DIA   = 0.16
    X_HORA  = 0.16
    W_HORA  = 0.13
    X_DESC  = 0.29
    W_DESC  = 0.48
    X_POLO  = 0.77
    W_POLO  = 0.23

    slot_idx = 0

    for fatia in fatias:
        data_dt  = fatia.data_dt
        grp      = fatia.df.reset_index(drop=True)
        n_ativ   = len(grp)
        dia_nome = DIAS_PT.get(data_dt.weekday(), "")
        dia_data = data_dt.strftime("%d/%m")

        # ------------------------------------------------------------------
        # Coluna lateral do dia
        # ------------------------------------------------------------------
        y_top_dia    = 1.0 - slot_idx * slot_h
        altura_bloco = n_ativ * slot_h

        ax.add_patch(plt.Rectangle(
            (X_DIA, y_top_dia - altura_bloco),
            W_DIA, altura_bloco,
            facecolor=PALETA["day_col_bg"],
            edgecolor=PALETA["grid"],
            linewidth=0.8,
            transform=ax.transAxes,
            zorder=2,
        ))

        y_centro_dia = y_top_dia - altura_bloco / 2
        ax.text(X_DIA + W_DIA / 2, y_centro_dia + 0.008,
                dia_nome, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=15, fontweight="bold",
                color=PALETA["day_col_text"], zorder=3)
        ax.text(X_DIA + W_DIA / 2, y_centro_dia - 0.022,
                dia_data, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color="#A8C7DC", zorder=3)

        # Indicador "continua →" na borda inferior da faixa
        if fatia.has_continuation:
            ax.text(X_DIA + W_DIA / 2, y_top_dia - altura_bloco + slot_h * 0.18,
                    "cont. →", transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=8.5, color="#A8C7DC",
                    style="italic", zorder=4)

        # Indicador "← cont." na borda superior da faixa
        if fatia.is_continuation:
            ax.text(X_DIA + W_DIA / 2, y_top_dia - slot_h * 0.18,
                    "← cont.", transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=7.5, color="#A8C7DC",
                    style="italic", zorder=4)

        # ------------------------------------------------------------------
        # Linhas de atividade
        # ------------------------------------------------------------------
        for i, (_, row) in enumerate(grp.iterrows()):
            y_top  = 1.0 - (slot_idx + i) * slot_h
            y_bot  = y_top - slot_h
            y_mid  = (y_top + y_bot) / 2

            # Fundo branco uniforme + separadores finos
            ax.add_patch(plt.Rectangle(
                (X_HORA, y_bot), 1.0 - X_HORA, slot_h,
                facecolor=PALETA["cell_bg"], edgecolor="none",
                linewidth=0, transform=ax.transAxes,
            ))
            # Separador horizontal inferior de cada atividade:
            # começa em X_HORA para não cortar a coluna dia/data nas
            # atividades intermediárias do mesmo dia.
            eh_ultimo_do_dia = (i == n_ativ - 1)
            if eh_ultimo_do_dia:
                # Última atividade do dia: linha vai de ponta a ponta
                ax.axhline(y_bot, color=PALETA["grid"],
                           linewidth=1.4, xmin=0, xmax=1.0)
            else:
                # Atividades intermediárias: linha só nas colunas à direita do dia
                ax.axhline(y_bot, color=PALETA["grid"],
                           linewidth=0.8, xmin=X_HORA, xmax=1.0)

            # Separadores verticais entre colunas
            for _xv in [X_HORA, X_DESC, X_POLO]:
                ax.axvline(_xv, color=PALETA["grid"],
                           linewidth=0.8, ymin=y_bot, ymax=y_top)

            # Linha separadora superior do bloco de dia (início de novo dia)
            if i == 0:
                ax.axhline(y_top, color=PALETA["grid"],
                           linewidth=1.4, xmin=0, xmax=1)

            # ── Horário: lê coluna e exibe apenas hh:mm ──────────────────
            hora_raw = norm(row["HORA"])
            # 1) Pega só a parte antes de qualquer separador (–, -, —, /)
            hora_inicio = re.split(r"[\-–—/]", hora_raw)[0].strip()
            # 2) Extrai hh:mm descartando segundos se existirem
            _m = re.match(r"(\d{1,2}:\d{2})", hora_inicio)
            hora_inicio = _m.group(1) if _m else hora_inicio

            ax.text(X_HORA + W_HORA / 2, y_mid,
                    hora_inicio,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=PALETA["hour_text"])

            # ── Descrição — com quebra de linha automática ───────────────
            desc = norm(row["DESCRICAO"])
            polo = norm(row["POLO"])

            # Largura disponível para o texto da descrição em pixels
            _fig_aux   = fig if fig is not None else ax.get_figure()
            _fw, _fh   = _fig_aux.get_size_inches()
            _dpi       = _fig_aux.dpi
            _axes_w_px = _fw * _dpi * 0.96
            _desc_px   = W_DESC * _axes_w_px - 12   # margem de 12 px

            desc_linhas = _quebrar_polo_em_linhas(desc, 13, _desc_px,
                                                   _fw, _fh, _dpi)
            n_desc      = len(desc_linhas)
            # Espaçamento entre linhas como fração do slot
            line_gap    = slot_h * 0.28
            # Ponto de partida: centraliza o bloco de texto no terço superior
            y_desc_top  = y_mid + slot_h * 0.16 + (n_desc - 1) * line_gap / 2

            for k, linha_desc in enumerate(desc_linhas):
                ax.text(X_DESC + 0.008, y_desc_top - k * line_gap,
                        linha_desc,
                        transform=ax.transAxes,
                        ha="left", va="center",
                        fontsize=13, fontweight="bold",
                        color=PALETA["cell_text"],
                        clip_on=True)

            # ── Badge polo — quebra por palavra + redução de fonte se necessário
            if polo:
                _fig = fig if fig is not None else ax.get_figure()
                _renderizar_polo_badge(
                    ax, polo, X_POLO, W_POLO, y_mid, slot_h,
                    _fig.get_size_inches(), _fig.dpi
                )

            # ── Professor (linha de baixo) ───────────────────────────────
            prof = norm(row["PROFESSOR"])
            if prof:
                # Offset aumentado para 0.26 (era 0.18) para afastar da descrição
                # após o aumento de 2 pontos na fonte (11.5 → 13.5)
                ax.text(X_DESC + 0.008, y_mid - slot_h * 0.26,
                        f"Prof.: {prof}",
                        transform=ax.transAxes,
                        ha="left", va="center",
                        fontsize=14.5,
                        color=PALETA["prof_text"],
                        style="italic",
                        clip_on=True)

        slot_idx += n_ativ

    # Linha de fechamento inferior (na base da última atividade)
    ax.axhline(1.0 - slot_idx * slot_h,
               color=PALETA["grid"], linewidth=1.4, xmin=0, xmax=1)


def _desenhar_rodape(ax, legenda: str = ""):
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                color=PALETA["bg"],
                                transform=ax.transAxes))
    ax.text(0.03, 0.65, "CEAD – Centro de Educação Aberta e a Distância | UFPI",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=8, color=PALETA["muted"])
    if legenda:
        ax.text(0.03, 0.25, legenda,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7.5, color=PALETA["muted"])


# ══════════════════════════════════════════════════════════════════════════════
# 6) FUNÇÃO PRINCIPAL DE GERAÇÃO DE CARD
# ══════════════════════════════════════════════════════════════════════════════
def gerar_card(
    pagina_obj,
    semana_str: str,
    pagina: int,
    total_paginas: int,
    aspect: str,
    out_png: Path,
    modulo: str = "",
):
    # figsize grande para renderização legível; DPI calculado para 1080 px exatos
    if aspect == "9:16":
        fig_w, fig_h = 10.8, 19.2
        px_w,  px_h  = 1080, 1920
    else:
        fig_w, fig_h = 10.8, 13.5
        px_w,  px_h  = 1080, 1350

    dpi_saida = px_w / fig_w   # 100 dpi → 1080 px

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_saida)
    fig.patch.set_facecolor(PALETA["bg"])

    h_header    = H_HEADER
    h_colheader = H_COLHEADER
    h_footer    = H_FOOTER
    h_corpo     = H_CORPO

    # Safe zone: top=0.88 garante 12% de margem no topo evitando corte Instagram
    gs = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[h_header, h_colheader, h_corpo, h_footer],
        hspace=0.0,
        top=0.94, bottom=0.02, left=0.02, right=0.98,
    )

    ax_cab   = fig.add_subplot(gs[0])
    ax_hcol  = fig.add_subplot(gs[1])
    ax_corpo = fig.add_subplot(gs[2])
    ax_rod   = fig.add_subplot(gs[3])

    _desenhar_cabecalho(ax_cab, semana_str, pagina, total_paginas, modulo=modulo)
    _desenhar_header_colunas(ax_hcol)
    _desenhar_corpo(ax_corpo, pagina_obj.fatias, pagina_obj.n_ativ(), fig=fig)
    _desenhar_rodape(ax_rod)

    fig.savefig(out_png, dpi=dpi_saida)
    plt.close(fig)
    print(f"OK: {out_png.name}  ({px_w}x{px_h} px)")


# ══════════════════════════════════════════════════════════════════════════════
# 7) CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Gera cards PNG de Agenda Semanal para redes sociais."
    )
    ap.add_argument("--xlsx",    default="agenda.xlsx",
                    help="Arquivo Excel com as atividades da semana.")
    ap.add_argument("--aspect",  choices=["4:5", "9:16"], default="4:5",
                    help="Formato do card: 4:5 (feed) ou 9:16 (stories).")
    ap.add_argument("--outdir",  default="saida",
                    help="Pasta de saída para os PNGs.")
    ap.add_argument("--linhas-por-card", type=int, default=None,
                    dest="linhas_por_card",
                    help="Sobrescreve o cálculo automático de atividades por card.")
    args = ap.parse_args()

    df = carregar_excel(args.xlsx)
    if df.empty:
        raise SystemExit("Nenhuma atividade encontrada no arquivo Excel.")

    semana_str = obter_intervalo_semana(df)
    slot_h_ef  = 1.0 / args.linhas_por_card if args.linhas_por_card else SLOT_H

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Separação por módulo: se coluna MODULO preenchida, gera agenda por módulo.
    # Caso vazia/ausente, comportamento original (agenda única, sem badge).
    modulos = [m for m in df["MODULO"].unique() if m]
    if modulos:
        grupos = [(m, df[df["MODULO"] == m].reset_index(drop=True))
                  for m in sorted(modulos)]
    else:
        grupos = [("", df)]

    total_gerados = 0
    for modulo, df_mod in grupos:
        pages = paginar(df_mod, slot_h_ef, 1.0)
        total = len(pages)
        mod_slug = re.sub(r"[^A-Za-z0-9]", "_", modulo).lower() if modulo else ""
        prefix   = f"{mod_slug}_" if mod_slug else "agenda_semana_"
        if modulo:
            print(f"\n--- Gerando agenda: {modulo} ({total} card(s)) ---")
        for i, pag_obj in enumerate(pages, start=1):
            asp_slug = args.aspect.replace(":", "x")
            out_png  = outdir / f"{prefix}p{i:02d}_{asp_slug}.png"
            gerar_card(
                pagina_obj    = pag_obj,
                semana_str    = semana_str,
                pagina        = i,
                total_paginas = total,
                aspect        = args.aspect,
                out_png       = out_png,
                modulo        = modulo,
            )
        total_gerados += total

    print(f"\n{total_gerados} card(s) gerado(s) em '{outdir}/'")


if __name__ == "__main__":
    main()

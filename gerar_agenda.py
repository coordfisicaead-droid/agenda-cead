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
SLOT_H = 0.06   # <- AJUSTE AQUI

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
    "bg":               "#F7F8FA",
    "header_bg":        "#0B2A3A",
    "header_text":      "#FFFFFF",

    "block_title_bg":   "#123B52",
    "block_title_text": "#FFFFFF",

    "table_header_bg":  "#E9EEF5",
    "table_header_text":"#0E1726",

    "cell_bg":          "#FFFFFF",
    "cell_alt_bg":      "#F0F4F9",   # linha alternada
    "cell_text":        "#0E1726",

    "grid":             "#D7DDE5",

    "day_col_bg":       "#123B52",   # coluna lateral do dia
    "day_col_text":     "#FFFFFF",

    "badge_bg":         "#D6E4F0",   # badge de polo
    "badge_text":       "#0B2A3A",

    "hour_text":        "#1A5276",
    "prof_text":        "#5B6776",
    "muted":            "#5B6776",
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
def carregar_excel(path_xlsx: str) -> pd.DataFrame:
    """Lê o Excel ignorando a 1ª coluna e atribui nomes fixos às colunas."""
    raw = pd.read_excel(path_xlsx, header=0)

    # Garante que há colunas suficientes (preenche com NaN se faltar)
    while len(raw.columns) < 7:
        raw[f"_extra_{len(raw.columns)}"] = np.nan

    cols = raw.columns.tolist()
    df = pd.DataFrame()
    df["DATA"]      = raw.iloc[:, 1].apply(norm)
    df["HORA"]      = raw.iloc[:, 2].apply(norm)
    # Colunas 3 e 4 formam a descrição (col 3 primeiro, col 4 segundo)
    desc1           = raw.iloc[:, 3].apply(norm)
    desc2           = raw.iloc[:, 4].apply(norm)
    df["DESCRICAO"] = desc1.str.cat(desc2.where(desc2 != "", other=""), sep=" ", na_rep="").str.strip()
    df["POLO"]      = raw.iloc[:, 5].apply(norm)
    df["PROFESSOR"] = raw.iloc[:, 6].apply(norm)

    # Parse de data
    df["DATA_DT"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATA_DT"]).copy()

    # Ordenação: data → hora → polo
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
def _desenhar_cabecalho(ax, semana_str: str, pagina: int, total: int):
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                color=PALETA["header_bg"],
                                transform=ax.transAxes))

    # Linha 1 – título institucional
    ax.text(0.03, 0.76, "Licenciatura em Física - CEAD",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=20, fontweight="bold", color=PALETA["header_text"])

    # Linha 2 – subtítulo
    ax.text(0.03, 0.50, "Agenda Semanal",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=14, fontweight="bold", color=PALETA["header_text"])

    # Linha 3 – intervalo da semana + paginação
    pag_str = f"  •  Página {pagina} de {total}" if total > 1 else ""
    ax.text(0.03, 0.22, f"Semana {semana_str}{pag_str}",
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

    # Proporções das colunas — idênticas às usadas no corpo
    X_DIA  = 0.0;  W_DIA  = 0.12
    X_HORA = 0.12; W_HORA = 0.14
    X_DESC = 0.12 + 0.17; W_DESC = 0.52
    X_POLO = X_DESC + W_DESC; W_POLO = 1.0 - X_POLO

    labels = [
        ("DIA / DATA",             X_DIA  + W_DIA  / 2),
        ("HORÁRIO",                X_HORA + W_HORA / 2),
        ("DESCRIÇÃO / PROFESSOR",  X_DESC + W_DESC / 2),
        ("POLO",                   X_POLO + W_POLO / 2),
    ]

    for txt, x in labels:
        ax.text(x, 0.5, txt,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=PALETA["table_header_text"])

    # Linha inferior
    ax.axhline(0, color=PALETA["grid"], linewidth=1.2, xmin=0, xmax=1)


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
    Renderiza o badge do polo com:
      1. Tenta fonte padrão (9.5) em 1 linha
      2. Se não couber, quebra em 2 linhas por palavras completas
      3. Se ainda não couber em 2 linhas, reduz a fonte progressivamente (mín. 6.5)
    O badge usa a linha superior (y_mid + offset) e, quando necessário,
    a linha inferior (espaço do professor, sem conflito pois está na coluna direita).
    """
    fig_w_in, fig_h_in = fig_size_inches

    # Largura disponível na coluna polo em pixels
    # W_POLO é fração do axes; o axes ocupa left=0.02..right=0.98 da figura
    axes_w_px = fig_w_in * dpi * 0.96          # largura útil do axes em px
    col_px    = W_POLO * axes_w_px * 0.82      # margem interna do badge (~18%)

    fonte_base = 9.5
    fonte_min  = 6.5

    # ── Tenta encaixar numa ou duas linhas, reduzindo fonte se necessário ──
    fonte = fonte_base
    while fonte >= fonte_min:
        linhas = _quebrar_polo_em_linhas(polo, fonte, col_px,
                                          fig_w_in, fig_h_in, dpi)
        # Verifica se todas as linhas cabem
        todas_cabem = all(
            _medir_texto_px(l, fonte, fig_w_in, fig_h_in, dpi) <= col_px
            for l in linhas
        )
        if todas_cabem and len(linhas) <= 2:
            break
        fonte -= 0.5

    # Garante no máximo 2 linhas (une excesso na segunda se fonte chegou ao mín)
    if len(linhas) > 2:
        linhas = [linhas[0], " ".join(linhas[1:])]

    # ── Posições verticais ─────────────────────────────────────────────────
    if len(linhas) == 1:
        posicoes = [y_mid + slot_h * 0.14]
    else:
        posicoes = [y_mid + slot_h * 0.24, y_mid - slot_h * 0.08]

    pad = 0.22 if len(linhas) == 1 else 0.18

    for linha, y_pos in zip(linhas, posicoes):
        ax.text(
            X_POLO + W_POLO / 2, y_pos,
            linha,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=fonte, fontweight="bold",
            color=PALETA["badge_text"],
            clip_on=True,
            bbox=dict(
                boxstyle=f"round,pad={pad}",
                facecolor=PALETA["badge_bg"],
                edgecolor=PALETA["grid"],
                linewidth=0.7,
            ),
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
    W_DIA   = 0.12
    X_HORA  = W_DIA
    W_HORA  = 0.14
    X_DESC  = X_HORA + W_HORA
    W_DESC  = 0.52
    X_POLO  = X_DESC + W_DESC
    W_POLO  = 1.0 - X_POLO

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
        ax.text(X_DIA + W_DIA / 2, y_centro_dia + 0.006,
                dia_nome, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=11.5, fontweight="bold",
                color=PALETA["day_col_text"], zorder=3)
        ax.text(X_DIA + W_DIA / 2, y_centro_dia - 0.015,
                dia_data, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10.5, color="#A8C7DC", zorder=3)

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

            # Fundo alternado — usa índice global para nunca repetir cor entre dias
            bg = PALETA["cell_bg"] if (slot_idx + i) % 2 == 0 else PALETA["cell_alt_bg"]
            ax.add_patch(plt.Rectangle(
                (X_HORA, y_bot), 1.0 - X_HORA, slot_h,
                facecolor=bg, edgecolor=PALETA["grid"],
                linewidth=0.6, transform=ax.transAxes,
            ))

            # Linha separadora superior do bloco de dia
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

            ax.text(X_HORA + W_HORA / 2, y_mid + slot_h * 0.14,
                    hora_inicio,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=10.5, fontweight="bold",
                    color=PALETA["hour_text"])

            # ── Descrição (linha de cima) ─────────────────────────────────
            desc = norm(row["DESCRICAO"])
            polo = norm(row["POLO"])

            ax.text(X_DESC + 0.008, y_mid + slot_h * 0.14,
                    desc,
                    transform=ax.transAxes,
                    ha="left", va="center",
                    fontsize=10.5, fontweight="bold",
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
                ax.text(X_DESC + 0.008, y_mid - slot_h * 0.15,
                        f"Prof.: {prof}",
                        transform=ax.transAxes,
                        ha="left", va="center",
                        fontsize=11.5,
                        color=PALETA["cell_text"],
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
):
    if aspect == "9:16":
        fig_w, fig_h = 10.8, 19.2
    else:                          # 4:5 (padrão feed)
        fig_w, fig_h = 10.8, 13.5

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor(PALETA["bg"])

    # ── Proporções verticais — usam as constantes globais (mesmas da paginação)
    h_header    = H_HEADER
    h_colheader = H_COLHEADER
    h_footer    = H_FOOTER
    h_corpo     = H_CORPO

    gs = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[h_header, h_colheader, h_corpo, h_footer],
        hspace=0.0,
        top=0.97, bottom=0.02, left=0.02, right=0.98,
    )

    ax_cab   = fig.add_subplot(gs[0])
    ax_hcol  = fig.add_subplot(gs[1])
    ax_corpo = fig.add_subplot(gs[2])
    ax_rod   = fig.add_subplot(gs[3])

    _desenhar_cabecalho(ax_cab, semana_str, pagina, total_paginas)
    _desenhar_header_colunas(ax_hcol)
    _desenhar_corpo(ax_corpo, pagina_obj.fatias, pagina_obj.n_ativ(), fig=fig)
    _desenhar_rodape(ax_rod)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"OK: {out_png.name}")


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

    semana_str    = obter_intervalo_semana(df)
    # slot_h pode ser sobrescrito via --linhas-por-card (converte N linhas em altura)
    slot_h_ef = 1.0 / args.linhas_por_card if args.linhas_por_card else SLOT_H
    pages     = paginar(df, slot_h_ef, 1.0)
    total      = len(pages)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, pag_obj in enumerate(pages, start=1):
        asp_slug = args.aspect.replace(":", "x")
        out_png  = outdir / f"agenda_semana_p{i:02d}_{asp_slug}.png"

        gerar_card(
            pagina_obj     = pag_obj,
            semana_str     = semana_str,
            pagina         = i,
            total_paginas  = total,
            aspect         = args.aspect,
            out_png        = out_png,
        )

    print(f"\n{total} card(s) gerado(s) em '{outdir}/'")


if __name__ == "__main__":
    main()

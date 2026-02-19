# app.py
# Streamlit — Transações por Hora (CSV do dia atual)
# Abas:
# - Principal: cards + gráfico (Todas/Empresa) + enviar 1 gráfico (Discord)
# - Top 10: 10 gráficos individuais (top 10 por total do dia) + enviar 10 gráficos (Discord)
#
# ✅ Ajustes:
# - Parser robusto de TOTAL TRANSACTIONS (não converte 44.0 em 440)
# - Validação cedo (alerta se houver formatação suspeita)
# - Labels (número exato) em cima de cada bolinha em todos os gráficos
#
# Secrets:
#   .streamlit/secrets.toml
#   DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/...."
#
# Requisitos:
#   pip install streamlit pandas matplotlib requests

import os
import glob
import io
import json
import re
from datetime import date, datetime

import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st


# =========================
# CONFIG PADRÃO
# =========================
DEFAULT_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
CSV_GLOB_PATTERN = "*.csv"

COL_HOUR = "Hora Transacao"
COL_ACCOUNT = "Movement Account ID"
COL_PERSON = "Person Name"
COL_TOTAL = "TOTAL TRANSACTIONS"

DEFAULT_COMPANIES = [
    "Pix na Hora",
    "MarchaPay",
    "VCONSULTING",
    "QUANTUM PAYMENTS",
    "Pagamento Seguro LTDA",
    "Dom Digital",
    "Winn Pay",
    "KP SERVICOS DIGITAIS LTDA",
    "Otm Pagamentos",
    "Sync Pay",
    "Magic Pay LTDA",
    "Neopag Intermediações",
    "Masterfy Intermediações de Pagamentos LTDA",
    "BNT TECNOLOGIA",
    "NERES NEGOCIOS DIGITAIS LTDA",
    "Beehive",
    "APEXBET",
    "FRENDZ",
    "Vanessa Silva",
    "TECNOLOGIA E PAGAMENTOS KING LTDA",
    "CINQ PAY",
    "JLB Loterias Marketing e Serviços",
    "HP RECEBIVEIS",
    "Manda Pix",
    "BLOO",
    "APOSTARAIZ LTDA",
    "GRS EMPRESARIAL",
    "PAYNEO",
    "IG SISTEMAS",
    "SOLUÇÕES EM PAGAMENTOS",
    "FB OPERACOES E NEGOCIOS",
    "WITE RECUPERATION TECNOLOGIA IA LTDA",
]


# =========================
# HELPERS
# =========================
def find_latest_csv(folder: str, pattern: str) -> str | None:
    paths = glob.glob(os.path.join(folder, pattern))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def parse_int_smart(value) -> int:
    """
    Parser robusto para:
      - "44.0" -> 44
      - "44,0" -> 44
      - "79.753" -> 79753 (ponto como milhar)
      - "1.234,0" -> 1234
      - "1,234.0" (raro) -> 1234 (tenta inferir)
    """
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0

    # pt-BR com milhar '.' e decimal ',' -> 1.234,56
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):
        s2 = s.replace(".", "").replace(",", ".")
        return int(round(float(s2)))

    # só vírgula decimal -> 44,0 / 123,45
    if re.fullmatch(r"\d+(,\d+)?", s):
        return int(round(float(s.replace(",", "."))))

    # só ponto decimal ou inteiro -> 44.0 / 44 / 1234.56
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return int(round(float(s)))

    # fallback: limpa lixo, tenta inferir
    s3 = re.sub(r"[^\d,\.]", "", s)

    # se tem '.' e ',' assume pt-BR (milhar '.' decimal ',')
    if "." in s3 and "," in s3:
        s3 = s3.replace(".", "").replace(",", ".")
        return int(round(float(s3)))

    # só vírgula
    if "," in s3 and "." not in s3:
        return int(round(float(s3.replace(",", "."))))

    # só ponto
    if "." in s3:
        return int(round(float(s3)))

    # só dígitos
    s3 = re.sub(r"[^\d]", "", s3)
    return int(s3) if s3 else 0


def normalize_int_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_int_smart).astype("int64")


def read_csv_smart(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        if len(df.columns) == 1 and ";" in df.columns[0]:
            raise ValueError("Provável separador ';'")
        return df
    except Exception:
        return pd.read_csv(csv_path, sep=";")


def validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    missing = [c for c in [COL_HOUR, COL_ACCOUNT, COL_PERSON, COL_TOTAL] if c not in df.columns]
    if missing:
        errors.append(
            f"CSV não contém as colunas esperadas: {missing}. "
            f"Colunas encontradas: {list(df.columns)}"
        )
        return df.iloc[0:0].copy(), warnings, errors

    # Hora
    df[COL_HOUR] = pd.to_numeric(df[COL_HOUR], errors="coerce")
    invalid_hour = df[COL_HOUR].isna() | (df[COL_HOUR] < 0) | (df[COL_HOUR] > 23)
    if invalid_hour.any():
        warnings.append(f"Foram removidas {int(invalid_hour.sum())} linhas com Hora Transacao inválida.")
    df = df[~invalid_hour].copy()
    df[COL_HOUR] = df[COL_HOUR].astype(int)

    # Keep raw total for early checks
    raw_total = df[COL_TOTAL].astype(str).str.strip()

    # Total (robusto)
    df[COL_TOTAL] = normalize_int_series(df[COL_TOTAL])

    # Early validation (pegar erro cedo)
    # 1) Se houver muitos valores com ".0" ou ",0", verificar se o parse não "colou zero"
    mask_dec0 = raw_total.str.match(r"^\d+[.,]0+$")
    if mask_dec0.any():
        # se antes era 44.0 e depois virou 440, seria suspeito.
        # Com parser novo isso não deve acontecer; mas deixamos alerta se detectar padrão estranho.
        raw_examples = raw_total[mask_dec0].head(10).tolist()
        warnings.append(f"Detectado formato decimal tipo '.0'/',0' em TOTAL TRANSACTIONS (amostra: {raw_examples}). Parser robusto aplicado.")

    # 2) Valores absurdamente altos -> alerta (configurável)
    max_val = int(df[COL_TOTAL].max()) if len(df) else 0
    if max_val > 10_000_000:
        warnings.append("TOTAL TRANSACTIONS muito alto (>10.000.000). Verifique se o CSV está com formatação inesperada.")

    # Strings
    df[COL_PERSON] = df[COL_PERSON].astype(str).str.strip()
    df[COL_ACCOUNT] = df[COL_ACCOUNT].astype(str).str.strip()

    # Deduplicação por (hora, empresa, conta)
    before = len(df)
    df = df.groupby([COL_HOUR, COL_PERSON, COL_ACCOUNT], as_index=False)[COL_TOTAL].sum()
    after = len(df)
    if after < before:
        warnings.append(f"Havia duplicidades; consolidado por (hora, empresa, conta). Linhas: {before} → {after}.")

    return df, warnings, errors


def filter_companies_by_substring(df: pd.DataFrame, allow_list: list[str]) -> pd.DataFrame:
    allow = [x.strip().lower() for x in allow_list if x.strip()]
    if not allow:
        return df.iloc[0:0].copy()

    def ok(name: str) -> bool:
        n = name.lower()
        return any(a in n for a in allow)

    return df[df[COL_PERSON].apply(ok)].copy()


def build_pivot_hour_company(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index=COL_HOUR, columns=COL_PERSON, values=COL_TOTAL, aggfunc="sum", fill_value=0)
        .sort_index()
    )
    if pivot.empty:
        return pivot
    max_h = int(pivot.index.max())
    pivot = pivot.reindex(range(0, max_h + 1), fill_value=0)  # sem lacunas
    return pivot


def fmt_int_pt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_hour(h: int | None) -> str:
    return "—" if h is None else f"{str(h).zfill(2)}:00"


def pick_series(pivot: pd.DataFrame, selected: str) -> pd.Series:
    if pivot.empty:
        return pd.Series(dtype="int64")
    if selected == "Todas":
        return pivot.sum(axis=1)
    if selected in pivot.columns:
        return pivot[selected]
    return pd.Series([0] * len(pivot.index), index=pivot.index, dtype="int64")


def compute_metrics(series_by_hour: pd.Series) -> dict:
    today = date.today()
    if series_by_hour.empty:
        return {"date": today, "last_hour": None, "total_day": 0, "avg_hour": 0, "last_hour_total": 0}

    last_hour = int(series_by_hour.index.max())
    total_day = int(series_by_hour.sum())
    hours_available = last_hour + 1
    avg_hour = int(round(total_day / hours_available)) if hours_available > 0 else 0
    last_hour_total = int(series_by_hour.loc[last_hour])

    return {
        "date": today,
        "last_hour": last_hour,
        "total_day": total_day,
        "avg_hour": avg_hour,
        "last_hour_total": last_hour_total,
    }


def make_line_chart(series_by_hour: pd.Series, title: str, subtitle: str | None = None) -> bytes:
    """
    ✅ Anota o valor exato em cima de cada bolinha.
    """
    fig = plt.figure(figsize=(8.2, 3.2))
    ax = fig.add_subplot(111)

    if series_by_hour.empty:
        ax.set_title("Sem dados")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Transações")
    else:
        x = list(series_by_hour.index)
        y = list(series_by_hour.values)

        ax.plot(x, y, marker="o")
        ax.set_title(title)

        if subtitle:
            ax.text(0.01, 0.95, subtitle, transform=ax.transAxes, va="top")

        ax.set_xlabel("Hora")
        ax.set_ylabel("Transações")
        ax.set_xticks(x)

        max_y = max(y) if y else 0

        for xi, yi in zip(x, y):
            ax.annotate(
                f"{int(yi)}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylim(bottom=0, top=max_y * 1.15 + 1)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def discord_send_multi_images(
    webhook_url: str,
    title: str,
    description: str,
    images: list[tuple[str, bytes]],
    chunk_size: int = 5,
) -> tuple[bool, str]:
    """
    Envia múltiplas imagens via webhook, dividindo em lotes.
    images: lista de (filename, image_bytes)
    """
    if not webhook_url:
        return False, "Webhook não configurado (st.secrets)."

    for i in range(0, len(images), chunk_size):
        chunk = images[i:i + chunk_size]

        embeds = []
        for fname, _ in chunk:
            embeds.append({
                "title": fname.replace(".png", ""),
                "description": description if i == 0 else "",
                "image": {"url": f"attachment://{fname}"},
                "type": "rich",
            })

        payload = {"embeds": embeds}

        files = {
            "payload_json": (None, json.dumps(payload, ensure_ascii=False), "application/json"),
        }
        for fname, bts in chunk:
            files[fname] = (fname, bts, "image/png")

        try:
            r = requests.post(webhook_url, files=files, timeout=40)
            if not (200 <= r.status_code < 300):
                return False, f"Falhou (HTTP {r.status_code}): {r.text[:300]}"
        except Exception as e:
            return False, f"Erro ao enviar: {e}"

    return True, f"Enviado em {((len(images) - 1) // chunk_size) + 1} mensagem(ns)."


# =========================
# UI
# =========================
st.set_page_config(page_title="Transações por Hora", layout="wide")
st.title("Transações por Hora — Monitoramento")

with st.sidebar:
    st.header("Fonte do CSV")
    folder = st.text_input("Pasta dos CSVs", value=DEFAULT_DOWNLOAD_DIR)
    pattern = st.text_input("Padrão (glob)", value=CSV_GLOB_PATTERN)

    st.header("Empresas monitoradas")
    companies_text = st.text_area(
        "Lista fixa (1 por linha)",
        value="\n".join(DEFAULT_COMPANIES),
        height=260,
    )
    allow_list = [x.strip() for x in companies_text.splitlines() if x.strip()]

    st.header("Discord (secrets)")
    webhook_url = st.secrets.get("DISCORD_WEBHOOK_URL", "").strip()
    st.text_input("Webhook status", value=("configurado" if webhook_url else "não configurado"), disabled=True)

    st.header("Atualização")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)

if auto_refresh:
    try:
        st.autorefresh(interval=60_000, key="autorefresh_60s")
    except Exception:
        pass

latest = find_latest_csv(folder, pattern)
if not latest:
    st.warning(f"Nenhum CSV encontrado em: {folder} com padrão: {pattern}")
    st.stop()

try:
    df_raw = read_csv_smart(latest)
    df_clean, warns, errs = validate_and_prepare(df_raw)
except Exception as e:
    st.error(f"Erro lendo/validando CSV: {e}")
    st.stop()

if errs:
    for e in errs:
        st.error(e)
    st.stop()

for w in warns:
    st.warning(w)

df = filter_companies_by_substring(df_clean, allow_list)
if df.empty:
    st.error("Nenhuma linha do CSV bateu com a lista de empresas monitoradas.")
    sample_names = df_clean[COL_PERSON].dropna().unique().tolist()[:25]
    if sample_names:
        st.write("Exemplos de 'Person Name' no CSV (amostra):")
        st.write(sample_names)
    st.stop()

pivot = build_pivot_hour_company(df)

hdr_left, hdr_right = st.columns([2, 1])
with hdr_left:
    st.caption("Arquivo em uso")
    st.code(os.path.basename(latest), language="text")
with hdr_right:
    mtime = datetime.fromtimestamp(os.path.getmtime(latest))
    st.caption(f"Modificado em: {mtime.strftime('%d/%m/%Y %H:%M:%S')}")

tab_main, tab_top10 = st.tabs(["Principal", "Top 10 (Gráficos individuais)"])

with tab_main:
    left, right = st.columns([2, 1])
    with left:
        options = ["Todas"] + (list(pivot.columns) if not pivot.empty else [])
        selected = st.selectbox("Empresa", options=options, index=0, key="empresa_main")
    with right:
        st.caption("Dica")
        st.write("Selecione uma empresa para ver métricas e gráfico dessa empresa.")

    series = pick_series(pivot, selected)
    metrics = compute_metrics(series)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Data / Última hora disponível", metrics["date"].strftime("%d/%m/%Y"), fmt_hour(metrics["last_hour"]))
    with c2:
        st.metric("Total do dia (00:00 → última hora)", fmt_int_pt(metrics["total_day"]))
    with c3:
        st.metric("Média por hora (dia)", fmt_int_pt(metrics["avg_hour"]))
    with c4:
        st.metric("Transações na última hora", fmt_int_pt(metrics["last_hour_total"]))

    title = "Transações por Hora — Total (empresas filtradas)" if selected == "Todas" else f"Transações por Hora — {selected}"
    img_bytes = make_line_chart(series, title)
    st.image(img_bytes, caption="Transações por hora (linha)", use_container_width=True)

    st.divider()
    st.subheader("Enviar para Discord (visão atual)")
    send_disabled = (not webhook_url) or series.empty
    send = st.button("Enviar resumo + gráfico", type="primary", disabled=send_disabled, key="send_main")

    if send:
        janela = f"{metrics['date'].strftime('%d/%m/%Y')} — última hora: {fmt_hour(metrics['last_hour'])}"
        desc = (
            f"**Janela:** {janela}\n"
            f"**Empresa:** {selected}\n"
            f"**Total do dia:** {fmt_int_pt(metrics['total_day'])}\n"
            f"**Média/hora:** {fmt_int_pt(metrics['avg_hour'])}\n"
            f"**Última hora:** {fmt_int_pt(metrics['last_hour_total'])}\n"
            f"**Fonte:** {os.path.basename(latest)}"
        )
        ok, msg = discord_send_multi_images(
            webhook_url=webhook_url,
            title="Monitoramento — Transações por Hora",
            description=desc,
            images=[("grafico.png", img_bytes)],
            chunk_size=1,
        )
        st.success(msg) if ok else st.error(msg)

with tab_top10:
    st.subheader("Top 10 empresas (filtradas) — gráficos individuais")

    total_by_company = pivot.sum(axis=0).sort_values(ascending=False)
    top10 = total_by_company.head(10)

    if top10.empty:
        st.info("Sem dados para Top 10.")
        st.stop()

    cols = st.columns(2)
    images_to_send: list[tuple[str, bytes]] = []
    last_hour_global = int(pivot.index.max()) if not pivot.empty else None

    for idx, (company, total) in enumerate(top10.items(), start=1):
        series_c = pivot[company]
        subtitle = f"Total do dia: {fmt_int_pt(int(total))} | Última hora: {fmt_hour(last_hour_global)}"
        img_c = make_line_chart(series_c, title=f"{idx}) {company}", subtitle=subtitle)

        images_to_send.append((f"{idx:02d} - {company}.png", img_c))

        target_col = cols[idx % 2]
        with target_col:
            st.image(img_c, caption=f"{company} — Total: {fmt_int_pt(int(total))}", use_container_width=True)

    st.divider()
    st.subheader("Enviar Top 10 para Discord")

    ranking_lines = [f"{i}. {company} — **{fmt_int_pt(int(total))}**" for i, (company, total) in enumerate(top10.items(), start=1)]
    desc = (
        f"**Data:** {date.today().strftime('%d/%m/%Y')}\n"
        f"**Última hora disponível:** {fmt_hour(last_hour_global)}\n\n"
        f"**Ranking (Total do dia):**\n" + "\n".join(ranking_lines)
    )

    send_top10_disabled = (not webhook_url) or (len(images_to_send) == 0)
    send_top10 = st.button("Enviar Top 10 (10 gráficos)", type="primary", disabled=send_top10_disabled, key="send_top10")

    if send_top10:
        ok, msg = discord_send_multi_images(
            webhook_url=webhook_url,
            title="Top 10 — Transações por Hora",
            description=desc,
            images=images_to_send,
            chunk_size=5,  # manda 5 + 5 por segurança
        )
        st.success(msg) if ok else st.error(msg)
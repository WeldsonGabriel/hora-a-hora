# app.py
# Streamlit — Transações por Hora (CSV do dia atual) via UPLOAD
# Abas:
# - Principal: cards + gráfico (Todas/Empresa) + enviar 1 gráfico (Discord)
# - Top 10: 10 gráficos individuais (top 10 por total do dia) + enviar 10 gráficos (Discord)
#
# ✅ Fluxo:
# 1) Usuário faz upload do CSV
# 2) Clica "Processar CSV"
# 3) App usa os dados processados (session_state)
#
# ✅ Ajustes:
# - Parser robusto de TOTAL TRANSACTIONS (não converte 44.0 em 440)
# - Validação cedo
# - Labels (número exato) em cima de cada bolinha em todos os gráficos
#
# Secrets:
#   .streamlit/secrets.toml
#   DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/...."
#
# Requisitos:
#   pip install streamlit pandas matplotlib requests

import io
import json
import re
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st


# =========================
# CONFIG
# =========================
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
# PARSE ROBUSTO
# =========================
def parse_int_smart(value) -> int:
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

    # fallback: limpa lixo e tenta inferir
    s3 = re.sub(r"[^\d,\.]", "", s)

    if "." in s3 and "," in s3:
        s3 = s3.replace(".", "").replace(",", ".")
        return int(round(float(s3)))

    if "," in s3 and "." not in s3:
        return int(round(float(s3.replace(",", "."))))

    if "." in s3:
        return int(round(float(s3)))

    s3 = re.sub(r"[^\d]", "", s3)
    return int(s3) if s3 else 0


def normalize_int_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_int_smart).astype("int64")


def read_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Lê CSV a partir do upload.
    Tenta ',' e se der ruim tenta ';'.
    """
    raw = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep=";")


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

    # Keep raw totals for early checks
    raw_total = df[COL_TOTAL].astype(str).str.strip()

    # Total robusto
    df[COL_TOTAL] = normalize_int_series(df[COL_TOTAL])

    # Validação cedo
    mask_dec0 = raw_total.str.match(r"^\d+[.,]0+$")
    if mask_dec0.any():
        examples = raw_total[mask_dec0].head(10).tolist()
        warnings.append(f"Detectado formato decimal tipo '.0'/',0' em TOTAL TRANSACTIONS (amostra: {examples}). Parser robusto aplicado.")

    max_val = int(df[COL_TOTAL].max()) if len(df) else 0
    if max_val > 10_000_000:
        warnings.append("TOTAL TRANSACTIONS muito alto (>10.000.000). Verifique se o CSV está com formatação inesperada.")

    df[COL_PERSON] = df[COL_PERSON].astype(str).str.strip()
    df[COL_ACCOUNT] = df[COL_ACCOUNT].astype(str).str.strip()

    # Consolidar duplicados
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
st.title("Transações por Hora — Monitoramento (Upload)")

webhook_url = st.secrets.get("DISCORD_WEBHOOK_URL", "").strip()

with st.sidebar:
    st.header("CSV (Upload)")
    uploaded = st.file_uploader("Selecione o CSV", type=["csv"])

    st.header("Empresas monitoradas")
    companies_text = st.text_area(
        "Lista fixa (1 por linha)",
        value="\n".join(DEFAULT_COMPANIES),
        height=260,
    )
    allow_list = [x.strip() for x in companies_text.splitlines() if x.strip()]

    st.header("Ação")
    process = st.button("Processar CSV", type="primary", disabled=(uploaded is None))

    st.header("Discord (secrets)")
    st.text_input("Webhook status", value=("configurado" if webhook_url else "não configurado"), disabled=True)

# session state init
if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "pivot" not in st.session_state:
    st.session_state["pivot"] = pd.DataFrame()
if "df_filtered" not in st.session_state:
    st.session_state["df_filtered"] = pd.DataFrame()
if "filename" not in st.session_state:
    st.session_state["filename"] = None
if "warnings" not in st.session_state:
    st.session_state["warnings"] = []
if "errors" not in st.session_state:
    st.session_state["errors"] = []

# Process button
if process and uploaded is not None:
    try:
        df_raw = read_csv_from_upload(uploaded)
        df_clean, warns, errs = validate_and_prepare(df_raw)
        df_filtered = filter_companies_by_substring(df_clean, allow_list)

        if df_filtered.empty:
            st.session_state["processed"] = False
            st.session_state["pivot"] = pd.DataFrame()
            st.session_state["df_filtered"] = pd.DataFrame()
            st.session_state["filename"] = uploaded.name
            st.session_state["warnings"] = warns
            st.session_state["errors"] = ["Nenhuma linha do CSV bateu com a lista de empresas monitoradas."]
        else:
            pivot = build_pivot_hour_company(df_filtered)
            st.session_state["processed"] = True
            st.session_state["pivot"] = pivot
            st.session_state["df_filtered"] = df_filtered
            st.session_state["filename"] = uploaded.name
            st.session_state["warnings"] = warns
            st.session_state["errors"] = errs
    except Exception as e:
        st.session_state["processed"] = False
        st.session_state["pivot"] = pd.DataFrame()
        st.session_state["df_filtered"] = pd.DataFrame()
        st.session_state["filename"] = uploaded.name if uploaded else None
        st.session_state["warnings"] = []
        st.session_state["errors"] = [f"Erro lendo/validando CSV: {e}"]

# Show status
if not st.session_state["processed"]:
    if uploaded is None:
        st.info("Faça upload do CSV e clique em **Processar CSV**.")
        st.stop()

    # houve upload mas ainda não processou ou deu erro
    for w in st.session_state["warnings"]:
        st.warning(w)
    for e in st.session_state["errors"]:
        st.error(e)

    if st.session_state["filename"]:
        st.caption(f"Arquivo selecionado: {st.session_state['filename']}")

    if st.session_state["errors"]:
        st.stop()

    st.info("Clique em **Processar CSV** para gerar os gráficos.")
    st.stop()

# Processed OK
for w in st.session_state["warnings"]:
    st.warning(w)

pivot = st.session_state["pivot"]
filename = st.session_state["filename"] or "CSV"

st.caption(f"CSV processado: **{filename}**")

tab_main, tab_top10 = st.tabs(["Principal", "Top 10 (Gráficos individuais)"])

with tab_main:
    options = ["Todas"] + (list(pivot.columns) if not pivot.empty else [])
    selected = st.selectbox("Empresa", options=options, index=0, key="empresa_main")

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

    if not webhook_url:
        st.info("Webhook não configurado. Defina DISCORD_WEBHOOK_URL em .streamlit/secrets.toml.")

    if send:
        janela = f"{metrics['date'].strftime('%d/%m/%Y')} — última hora: {fmt_hour(metrics['last_hour'])}"
        desc = (
            f"**Janela:** {janela}\n"
            f"**Empresa:** {selected}\n"
            f"**Total do dia:** {fmt_int_pt(metrics['total_day'])}\n"
            f"**Média/hora:** {fmt_int_pt(metrics['avg_hour'])}\n"
            f"**Última hora:** {fmt_int_pt(metrics['last_hour_total'])}\n"
            f"**Fonte:** {filename}"
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
            chunk_size=5,
        )
        st.success(msg) if ok else st.error(msg)

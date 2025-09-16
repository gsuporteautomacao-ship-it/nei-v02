import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LLM (opcional via Hugging Face)
HF_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# JSONPath (somente para o modo JSON)
try:
    from jsonpath_ng import parse as jsonpath_parse
    HAS_JSONPATH = True
except Exception:
    HAS_JSONPATH = False

st.set_page_config(page_title="JSON & Tabela Q&A Bot (LLM opcional)", page_icon="🧠", layout="wide")


# -------------------- Utilidades comuns --------------------

def norm_txt(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def make_row_text(row: pd.Series) -> str:
    parts = []
    for c, v in row.items():
        sv = norm_txt(v).strip()
        if sv:
            parts.append(f"{c}: {sv}")
    return " | ".join(parts)

def build_tfidf(corpus: List[str]):
    if not corpus:
        return None, None
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_df=0.95, min_df=1)
    mat = vec.fit_transform(corpus)
    return vec, mat

def cosine_topk(vec, mat, query: str, top_k: int = 10):
    if vec is None or mat is None:
        return np.array([]), np.array([])
    q = vec.transform([query])
    sims = cosine_similarity(q, mat).ravel()
    idx = np.argsort(-sims)[:top_k]
    return idx, sims[idx]

def extract_numbers_units(q: str):
    # números + unidades comuns em automação
    units_map = {
        "nm": ["nm", "n.m", "n·m", "newton", "newtonmetro", "newton-metro", "newton meter", "newton meters"],
        "rpm": ["rpm", "rotações", "rotacoes"],
        "w": ["w", "watt", "watts"],
        "kw": ["kw", "kilowatt", "kilowatts"],
        "v": ["v", "vac", "vdc", "volts", "volt", "tensao", "tensão", "voltagem"],
        "a": ["a", "amp", "amps", "amper", "amperes", "corrente"],
        "mm": ["mm", "milimetro", "milímetro", "milimetros", "milímetros"],
        "cm": ["cm"],
        "m": [" m"," metro"],
        "hp": ["hp", "cv"],
    }
    q_clean = re.sub(r"[^\w\s\.,%-/°µμ]", " ", q.lower())
    nums = re.findall(r"\d+(?:[\.,]\d+)?", q_clean)
    tokens = re.findall(r"[a-zµμ°/%]+", q_clean)
    tokens = [t.replace("µ","u").replace("μ","u") for t in tokens]
    unit_hits = set()
    for u, aliases in units_map.items():
        for a in aliases:
            if a in tokens or a in q_clean:
                unit_hits.add(u)
                break
    return nums, list(unit_hits)

def guess_relevant_columns(df: pd.DataFrame, units: List[str]) -> List[str]:
    if df.empty:
        return []
    cols = [c for c in df.columns if isinstance(c, str)]
    score = {c:0 for c in cols}
    key_map = {
        "nm": ["nm","n.m","n·m","torque","par","newton"],
        "rpm":["rpm","veloc","rota","speed"],
        "w":["w","watt","pot","power"],
        "kw":["kw","kilowatt","pot","power"],
        "v":["v","volt","vac","vdc","tens","voltag"],
        "a":["a","amp","amper","corrente"],
        "mm":["mm","milim"],
        "cm":["cm"],
        "m":[" m","metro"],
        "hp":["hp","cv"],
    }
    for c in cols:
        lc = c.lower()
        for u in units:
            for kw in key_map.get(u, []):
                if kw in lc:
                    score[c] += 2
        # “conjunto/kit/modelo…” também contam
        for kw in ["conjunto","kit","modelo","partnumber","part number","pn","código","codigo","descr","produto","sku"]:
            if kw in lc:
                score[c] += 1
    ranked = [c for c,_ in sorted(score.items(), key=lambda kv: -kv[1]) if _>0]
    return ranked[:6]

def unit_number_regex(num: str, unit: str):
    # acha "20 Nm", "20Nm", "20 N.m", etc.
    ualt = {
        "nm": r"(?:n\.?m|nm|n·m|newton(?:-|\s*)metro|newton(?:\s*)meters?)",
        "rpm": r"(?:rpm)",
        "w": r"(?:w|watt[s]?)",
        "kw": r"(?:kw|kilowatt[s]?)",
        "v": r"(?:v|vac|vdc|volt[s]?)",
        "a": r"(?:a|amp[s]?|ampere[s]?)",
        "mm": r"(?:mm)",
        "cm": r"(?:cm)",
        "m": r"(?:\bm\b|metro[s]?)",
        "hp": r"(?:hp|cv)",
    }
    patt_u = ualt.get(unit, re.escape(unit))
    n = re.escape(num).replace("\,", "[\.,]")
    return re.compile(rf"\b{n}\s*{patt_u}\b", re.IGNORECASE)

def smart_boost_table(query: str, df: pd.DataFrame, row_texts: List[str]) -> np.ndarray:
    # boost para número+unidade e palavras “conjunto/kit”
    nums, units = extract_numbers_units(query)
    boosts = np.zeros(len(row_texts), dtype=float)
    if not row_texts:
        return boosts

    # termos de intenção
    kw_boost_terms = []
    if re.search(r"\bconjunto\b", query.lower()): kw_boost_terms.append("conjunto")
    if re.search(r"\bkit\b", query.lower()): kw_boost_terms.append("kit")
    if kw_boost_terms:
        for i, txt in enumerate(row_texts):
            for w in kw_boost_terms:
                if w in txt.lower():
                    boosts[i] += 0.15

    # número + unidade no mesmo trecho
    if nums and units:
        regs = [unit_number_regex(n, u) for n in nums for u in units]
        for i, txt in enumerate(row_texts):
            for rg in regs:
                if rg.search(txt):
                    boosts[i] += 0.5
                    break

    # cabeçalhos prováveis da unidade
    if units:
        cols = guess_relevant_columns(df, units)
        if cols:
            for i, row in enumerate(df[cols].fillna("").astype(str).apply(" | ".join, axis=1)):
                if any(n in row for n in nums) or any(u in row.lower() for u in units):
                    boosts[i] += 0.15

    # só número
    if nums and not units:
        for i, txt in enumerate(row_texts):
            if any(re.search(rf"\b{re.escape(n)}\b", txt) for n in nums):
                boosts[i] += 0.1

    return boosts

SYSTEM_PROMPT = """Você é um assistente que responde ESTRITAMENTE com base no contexto de linhas/colunas de uma TABELA ou de um JSON.
- Se a informação não estiver claramente no contexto, responda: "Não encontrei essa informação nos dados fornecidos."
- Ao citar, inclua o nome da coluna e/ou o índice da linha quando possível.
- Seja objetivo e responda em português do Brasil.
"""

def build_llm_prompt_from_rows(question: str, df: pd.DataFrame, row_indices: List[int], max_cells: int = 140) -> str:
    lines = []
    for ridx in row_indices:
        if ridx < 0 or ridx >= len(df):
            continue
        row = df.iloc[ridx]
        parts = []
        for c, v in row.items():
            vs = norm_txt(v)
            if vs:
                parts.append(f"{c}={vs}")
        row_txt = "; ".join(parts)
        if len(row_txt) > 2000:
            row_txt = row_txt[:2000] + "…"
        lines.append(f"[linha {ridx}] {row_txt}")
    context = "\n".join(lines) if lines else "(vazio)"
    prompt = f"""{SYSTEM_PROMPT}

# CONTEXTO (linhas candidatas)
{context}

# PERGUNTA
{question}

# INSTRUÇÕES
- Responda somente com base no CONTEXTO.
- Se houver várias opções, liste as mais relevantes com coluna/linha.
- Se não houver informação suficiente, diga que não foi encontrado.
"""
    return prompt

def call_hf_llm(model_id: str, prompt: str, token: str, max_new_tokens: int = 300, temperature: float = 0.2, timeout: int = 60):
    if not HF_AVAILABLE:
        return False, "Pacote huggingface_hub não está disponível."
    try:
        client = InferenceClient(model=model_id, token=token, timeout=timeout)
        output = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0.01,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        return True, output.strip()
    except Exception as e:
        return False, f"Falha ao chamar o modelo Hugging Face: {e}"


# -------------------- UI: fonte de dados --------------------

st.title("🧠 Q&A para JSON e Tabelas (com LLM opcional)")

mode = st.radio("Selecione a fonte de dados:", ["Tabela (Excel/CSV)", "JSON"], index=0)

with st.sidebar:
    st.header("Configuração")

    st.subheader("LLM (Hugging Face)")
    use_llm = st.toggle("Usar LLM para responder", value=True)
    model_id = st.text_input("Model ID", value="meta-llama/Meta-Llama-3.1-8B-Instruct")
    hf_token = st.text_input("HF Token", type="password", help="Cole aqui ou configure em st.secrets['HF_TOKEN']")
    if not hf_token and "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
        st.caption("Usando token de `st.secrets['HF_TOKEN']`.")


# -------------------- Modo TABELA --------------------
if mode == "Tabela (Excel/CSV)":
    st.subheader("Carregar tabela")
    file = st.file_uploader("Envie um .xlsx, .xls ou .csv", type=["xlsx","xls","csv"])
    example_btn = st.button("Carregar exemplo de tabela")

    df = None
    if example_btn:
        df = pd.DataFrame({
            "Produto": ["Conjunto Easy Servo 20 Nm", "Conjunto Easy Servo 12 Nm", "Servo Panasonic 400W", "Motor NEMA 23"],
            "Torque (Nm)": [20, 12, np.nan, np.nan],
            "Modelo": ["ES-MH342200 + ES-DH", "ES-MH342120 + ES-DH", "MBDLT25SF + MHMD042P1U", "57HS22"],
            "Descrição": ["Kit de motor de passo com encoder e drive (fechado)", "Kit 12 Nm", "Servo 400W sem freio", "Stepper NEMA 23"],
            "Código": ["NEI-SET-20NM", "NEI-SET-12NM", "PAN-400W", "N23-57HS22"]
        })
    elif file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            sheet = st.selectbox("Planilha", xls.sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheet)
    else:
        st.info("Envie sua planilha (.xlsx/.csv) ou clique em **Carregar exemplo de tabela**.")
        st.stop()

    if df is None or df.empty:
        st.error("Não foi possível carregar a tabela ou ela está vazia.")
        st.stop()

    st.success(f"Tabela carregada! {df.shape[0]} linhas × {df.shape[1]} colunas")
    with st.expander("Ver dados"):
        st.dataframe(df, use_container_width=True)

    cols = list(df.columns)
    title_col = st.selectbox("Coluna de título (opcional)", ["(auto)"] + cols, index=0)
    show_cols = st.multiselect("Colunas preferenciais para exibição", cols, default=cols[: min(6, len(cols))])

    row_texts = [make_row_text(row) for _, row in df.iterrows()]
    vec, mat = build_tfidf(row_texts)

    q = st.chat_input("Pergunte algo sobre a tabela… (ex.: 'Qual o conjunto de 20 Nm')")
    if q:
        idx, sims = cosine_topk(vec, mat, q, top_k=10)
        boosts = smart_boost_table(q, df, row_texts)
        if idx.size > 0:
            base_scores = np.zeros(len(df))
            base_scores[idx] = sims
            final_scores = base_scores + boosts
            topk = np.argsort(-final_scores)[:6]
        else:
            topk = np.argsort(-boosts)[:6]

        st.markdown("### Linhas mais relevantes")
        subset = df.iloc[topk][show_cols] if show_cols else df.iloc[topk]
        st.dataframe(subset, use_container_width=True)

        if use_llm and hf_token:
            prompt = build_llm_prompt_from_rows(q, df, list(topk))
            ok, out = call_hf_llm(model_id, prompt, hf_token, max_new_tokens=300, temperature=0.2)
            answer = out if ok else f"Não foi possível usar o LLM: {out}"
        else:
            lines = ["**Sugestões com base na tabela:**"]
            for ridx in topk:
                row = df.iloc[ridx]
                title_val = row.get(title_col, "") if title_col != "(auto)" else ""
                if not title_val:
                    for cand in ["Produto","Descrição","Modelo","Nome","Item"]:
                        if cand in df.columns:
                            title_val = norm_txt(row[cand]); break
                head = title_val or f"linha {ridx}"
                lines.append(f"- {head}")
            lines.append("\nDetalhe as colunas desejadas para uma resposta mais precisa.")
            answer = "\n".join(lines)

        with st.chat_message("assistant"):
            st.markdown(answer)

# -------------------- Modo JSON (compatibilidade) --------------------
else:
    st.subheader("Carregar JSON")
    uploaded = st.file_uploader("Arquivo .json", type=["json"])
    pasted = st.text_area("Ou cole JSON aqui", height=200, placeholder='{"empresa": {"nome": "Neoyama", "itens": [{"sku": "A6-400W", "preco": 1234.56}]}}')
    example_btn = st.button("Carregar exemplo JSON")

    DEFAULT_JSON = {
        "empresa": {
            "nome": "Exemplo S.A.",
            "departamentos": ["Vendas", "Compras", "TI"],
            "itens": [
                {"sku": "ABC-123", "descricao": "Motor NEMA 23", "preco": 299.9, "estoque": 12},
                {"sku": "A6-400W", "descricao": "Servo Panasonic 400W", "preco": 3899.0, "estoque": 3},
            ],
            "endereco": {"cidade": "Curitiba", "uf": "PR", "pais": "Brasil"},
            "ativo": True
        }
    }

    data_text = None
    if example_btn:
        data_text = json.dumps(DEFAULT_JSON, ensure_ascii=False, indent=2)
    elif uploaded is not None:
        data_text = uploaded.read().decode("utf-8", errors="ignore")
    elif pasted.strip():
        data_text = pasted

    if not data_text:
        st.info("Envie um JSON, cole o conteúdo ou carregue o exemplo.")
        st.stop()

    try:
        data_obj = json.loads(data_text)
    except Exception as e:
        st.error(f"JSON inválido: {e}")
        st.stop()

    def flatten_json(obj: Any, path: str = "") -> List[Dict[str, Any]]:
        rows = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                rows.extend(flatten_json(v, new_path))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_path = f"{path}[{i}]"
                rows.extend(flatten_json(v, new_path))
        else:
            vtype = type(obj).__name__
            try:
                vstr = json.dumps(obj, ensure_ascii=False)
            except Exception:
                vstr = str(obj)
            rows.append({"path": path, "type": vtype, "value_str": vstr, "value_raw": obj})
        return rows

    rows = flatten_json(data_obj)
    df = pd.DataFrame(rows)
    st.success(f"JSON carregado! {len(df)} valores de folha detectados.")
    with st.expander("Ver flatten (path → valor)"):
        st.dataframe(df[["path","type","value_str"]], use_container_width=True, hide_index=True)

    corpus = (df["path"].astype(str) + " :: " + df["value_str"].astype(str)).tolist()
    vec, mat = build_tfidf(corpus)

    with st.expander("🔎 JSONPath (opcional)"):
        if HAS_JSONPATH:
            jp_expr = st.text_input("Expressão JSONPath", value="")
            if jp_expr:
                from jsonpath_ng import parse as jsonpath_parse
                jp = jsonpath_parse(jp_expr)
                jp_res = [m.value for m in jp.find(data_obj)]
                st.write("Resultado JSONPath:")
                st.json(jp_res)
        else:
            st.info("Instale `jsonpath-ng` para usar JSONPath.")

    q = st.chat_input("Pergunte algo sobre o JSON…")
    if q:
        idx, sims = cosine_topk(vec, mat, q, top_k=8)
        hits = df.iloc[idx] if idx.size else df.iloc[:0]

        if use_llm and hf_token and idx.size:
            def limit_text(s, n=300): return (s if len(s)<=n else s[:n]+"…")
            context = "\n".join([f"- {row['path']} ({row['type']}): {limit_text(row['value_str'])}" for _, row in hits.iterrows()])
            prompt = f"""{SYSTEM_PROMPT}

# CONTEXTO
{context}

# PERGUNTA
{q}

# INSTRUÇÕES
- Responda com base apenas no CONTEXTO.
- Cite os `paths` quando possível.
"""
            ok, out = call_hf_llm(model_id, prompt, hf_token, max_new_tokens=300, temperature=0.2)
            answer = out if ok else f"Não foi possível usar o LLM: {out}"
        else:
            lines = []
            if hits.empty:
                lines.append("Não encontrei nada diretamente relacionado no JSON.")
            else:
                lines.append("Aqui está o que encontrei relacionado:")
                for _, row in hits.iterrows():
                    val = row["value_str"]
                    if len(val) <= 80 and "\n" not in val:
                        lines.append(f"- **{row['path']}** ({row['type']}): {val}")
                    else:
                        lines.append(f"- **{row['path']}** ({row['type']}):")
                        lines.append(f"```\n{val}\n```")
            answer = "\n".join(lines)

        with st.chat_message("assistant"):
            st.markdown(answer)

st.markdown("---")
st.caption("Dicas: no modo Tabela, perguntas como “Qual o conjunto de 20 Nm” ganham reforço por unidade/valor. No modo JSON, use JSONPath para filtros precisos.")

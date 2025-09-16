import json
from typing import Any, Dict, List, Tuple, Optional

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

try:
    from jsonpath_ng import parse as jsonpath_parse
    HAS_JSONPATH = True
except Exception:
    HAS_JSONPATH = False


st.set_page_config(page_title="JSON Q&A Bot (LLM)", page_icon="🧠", layout="wide")

# ---- Helpers ----

def flatten_json(obj: Any, path: str = "") -> List[Dict[str, Any]]:
    """
    Flatten nested JSON into rows with (path, type, value_str, value_raw).
    """
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
        # leaf
        vtype = type(obj).__name__
        try:
            vstr = json.dumps(obj, ensure_ascii=False)
        except Exception:
            vstr = str(obj)
        rows.append({"path": path, "type": vtype, "value_str": vstr, "value_raw": obj})
    return rows


def build_index(df: pd.DataFrame):
    """
    Build a TF-IDF index from path and value_str columns.
    """
    if df.empty:
        return None, None
    corpus = (df["path"].astype(str) + " :: " + df["value_str"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def search(df: pd.DataFrame, vectorizer: TfidfVectorizer, matrix, query: str, top_k: int = 8):
    """
    Return top_k matching rows with scores.
    """
    if df.empty or vectorizer is None or matrix is None:
        return pd.DataFrame(columns=["score", "path", "type", "value_str"])
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    idx = np.argsort(-sims)[:top_k]
    results = df.iloc[idx].copy()
    results.insert(0, "score", sims[idx])
    return results


def synthesize_answer_heu(question: str, hits: pd.DataFrame) -> str:
    """
    Heuristic answer synthesis without external LLMs.
    """
    if hits.empty:
        return "Não encontrei nada diretamente relacionado na base JSON. Tente reformular a pergunta ou use JSONPath."
    lines = []
    lines.append("Aqui está o que encontrei relacionado à sua pergunta:")
    for _, row in hits.iterrows():
        path = row["path"]
        vtype = row["type"]
        val = row["value_str"]
        # format small values inline; larger values in code block
        if len(val) <= 80 and "\\n" not in val:
            lines.append(f"- **{path}** ({vtype}): {val}")
        else:
            lines.append(f"- **{path}** ({vtype}):")
            lines.append(f"```\\n{val}\\n```")
    lines.append("Se precisar, refine a pergunta citando o caminho (path) desejado.")
    return "\\n".join(lines)


def limit_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"

SYSTEM_PROMPT = """Você é um assistente que responde ESTRITAMENTE com base no contexto extraído de um arquivo JSON.
- Se a informação não estiver claramente no contexto, responda: "Não encontrei essa informação no JSON fornecido."
- Cite os `paths` relevantes quando possível.
- Seja objetivo e responda em português do Brasil.
"""

def build_llm_prompt(question: str, hits_df: pd.DataFrame, jsonpath_results: Optional[List[Any]] = None) -> str:
    context_lines = []
    for _, row in hits_df.iterrows():
        path = row["path"]
        vtype = row["type"]
        val = limit_text(row["value_str"], 300)
        context_lines.append(f"- {path} ({vtype}): {val}")
    if jsonpath_results:
        try:
            import json as _json
            jp_text = _json.dumps(jsonpath_results, ensure_ascii=False)[:1500]
        except Exception:
            jp_text = str(jsonpath_results)[:1500]
        context_lines.append(f"- JSONPath_result: {jp_text}")
    context = "\\n".join(context_lines) if context_lines else "(vazio)"
    prompt = f"""{SYSTEM_PROMPT}

# CONTEXTO
{context}

# PERGUNTA
{question}

# INSTRUÇÕES DE RESPOSTA
- Responda com base apenas no CONTEXTO.
- Se apropriado, cite os caminhos entre crases, por exemplo `empresa.itens[0].sku`.
- Se a resposta não estiver no CONTEXTO, diga que não foi encontrada no JSON.
"""
    return prompt


def call_hf_llm(model_id: str, prompt: str, token: str, max_new_tokens: int = 400, temperature: float = 0.2, timeout: int = 60):
    try:
        client = InferenceClient(model=model_id, token=token, timeout=timeout)
        output = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        return True, output.strip()
    except Exception as e:
        return False, f"Falha ao chamar o modelo Hugging Face: {e}"


def run_jsonpath_query(data_obj: Any, expr: str) -> List[Any]:
    if not HAS_JSONPATH:
        return ["jsonpath-ng não está instalado. Ative-o no requirements.txt."]
    try:
        jp = jsonpath_parse(expr)
        return [match.value for match in jp.find(data_obj)]
    except Exception as e:
        return [f"Erro JSONPath: {e}"]


# ---- Sidebar ----

with st.sidebar:
    st.markdown("## Configuração")
    st.write("Carregue um arquivo JSON ou cole o conteúdo abaixo.")
    uploaded = st.file_uploader("Arquivo .json", type=["json"])
    pasted = st.text_area("Ou cole JSON aqui", height=200, placeholder='{"empresa": {"nome": "Neoyama", "itens": [{"sku": "A6-400W", "preco": 1234.56}]}}')
    example_btn = st.button("Carregar exemplo")

    st.markdown("---")
    st.markdown("## LLM (Hugging Face)")
    use_llm = st.toggle("Usar LLM para redigir respostas", value=True)
    default_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = st.text_input("Model ID", value=default_model, help="Exemplos: meta-llama/Meta-Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct, mistralai/Mixtral-8x7B-Instruct-v0.1")
    hf_token = st.text_input("Hugging Face API Token", type="password", help="Cole aqui ou configure em st.secrets['HF_TOKEN']")
    if not hf_token and "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
        st.caption("Usando token de `st.secrets['HF_TOKEN']`.")

# ---- Load data ----

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
    st.info("Use a barra lateral para enviar um arquivo JSON, colar conteúdo ou carregar o exemplo.")
    st.stop()

# Validate / parse JSON
try:
    data_obj = json.loads(data_text)
except Exception as e:
    st.error(f"JSON inválido: {e}")
    st.stop()

# Flatten
rows = flatten_json(data_obj)
df = pd.DataFrame(rows, columns=["path", "type", "value_str", "value_raw"])
st.success(f"JSON carregado! {len(df)} valores de folha detectados.")
with st.expander("Ver tabela flatten (path ➜ valor)"):
    st.dataframe(df[["path", "type", "value_str"]], use_container_width=True, hide_index=True)

# Build index
vectorizer, matrix = build_index(df)

# ---- Chat UI ----

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 Chatbot de JSON (com LLM opcional)")

st.caption("Faça perguntas em linguagem natural ou rode uma consulta JSONPath (ex: `$..itens[?(@.preco > 500)]`).")

# JSONPath box
jp_expr_val = ""
jp_res_cache: Optional[List[Any]] = None
with st.expander("🔎 JSONPath (opcional)"):
    if HAS_JSONPATH:
        jp_expr_val = st.text_input("Expressão JSONPath", value="")
        if jp_expr_val:
            jp_res_cache = run_jsonpath_query(data_obj, jp_expr_val)
            st.write("Resultado JSONPath:")
            st.json(jp_res_cache)
    else:
        st.info("Instale `jsonpath-ng` (já no requirements.txt).")

# Chat input
user_q = st.chat_input("Digite sua pergunta sobre o JSON…")
if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    # retrieve
    hits = search(df, vectorizer, matrix, user_q, top_k=8)

    # LLM or heuristic
    used_llm = False
    llm_answer = None
    context_sent = None

    if use_llm and hf_token:
        prompt = build_llm_prompt(user_q, hits, jsonpath_results=jp_res_cache)
        ok, out = call_hf_llm(model_id=model_id, prompt=prompt, token=hf_token, max_new_tokens=400, temperature=0.2)
        used_llm = ok
        llm_answer = out
        context_sent = prompt

    if used_llm:
        answer = llm_answer
    else:
        answer = synthesize_answer_heu(user_q, hits)
        if use_llm and not hf_token:
            answer += "\\n\\n> Observação: LLM não foi usado porque o token HF não foi configurado."
        elif use_llm and not HF_AVAILABLE:
            answer += "\\n\\n> Observação: Pacote `huggingface_hub` não disponível. Verifique `requirements.txt`."

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "hits": hits.to_dict(orient="records"),
        "used_llm": used_llm,
        "context": context_sent,
        "model_id": model_id if used_llm else None
    })

# Render conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            # Show top matches nicely
            if "hits" in msg and msg["hits"]:
                with st.expander("Ver trechos relevantes"):
                    hdf = pd.DataFrame(msg["hits"])
                    show_cols = ["score", "path", "type", "value_str"]
                    cols = [c for c in show_cols if c in hdf.columns]
                    st.dataframe(hdf[cols], use_container_width=True, hide_index=True)
            if msg.get("used_llm"):
                with st.expander(f"🔬 Contexto enviado ao LLM ({msg.get('model_id')})"):
                    st.code(msg.get("context") or "", language="markdown")
        else:
            st.markdown(msg["content"])

st.markdown("---")
st.caption("Dica: Para respostas mais ricas, ative o LLM na barra lateral e configure o token do Hugging Face.")
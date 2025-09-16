import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Optional LLM (Hugging Face) ----------
HF_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="NEI • Q&A Tabelas/JSON + Regra EASY SERVO", page_icon="🧠", layout="wide")

# ===================== Utils =====================
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
        for kw in ["conjunto","kit","modelo","partnumber","part number","pn","código","codigo","descr","produto","sku"]:
            if kw in lc:
                score[c] += 1
    ranked = [c for c,_ in sorted(score.items(), key=lambda kv: -kv[1]) if _>0]
    return ranked[:6]

def unit_number_regex(num: str, unit: str):
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
    n = re.escape(num).replace("\\,", "[\\.,]")
    return re.compile(rf"\b{n}\s*{patt_u}\b", re.IGNORECASE)

def smart_boost_table(query: str, df: pd.DataFrame, row_texts: List[str]) -> np.ndarray:
    nums, units = extract_numbers_units(query)
    boosts = np.zeros(len(row_texts), dtype=float)
    if not row_texts:
        return boosts

    # Intenção "conjunto/kit"
    kw_boost_terms = []
    if re.search(r"\bconjunto\b", query.lower()): kw_boost_terms.append("conjunto")
    if re.search(r"\bkit\b", query.lower()): kw_boost_terms.append("kit")
    if kw_boost_terms:
        for i, txt in enumerate(row_texts):
            for w in kw_boost_terms:
                if w in txt.lower():
                    boosts[i] += 0.15

    # número+unidade
    if nums and units:
        regs = [unit_number_regex(n, u) for n in nums for u in units]
        for i, txt in enumerate(row_texts):
            for rg in regs:
                if rg.search(txt):
                    boosts[i] += 0.5
                    break

    # colunas relevantes contendo números/unidades
    if units:
        cols = guess_relevant_columns(df, units)
        if cols:
            for i, row in enumerate(df[cols].fillna("").astype(str).apply(" | ", axis=1)):
                if any(n in row for n in nums) or any(u in row.lower() for u in units):
                    boosts[i] += 0.15

    # só número
    if nums and not units:
        for i, txt in enumerate(row_texts):
            if any(re.search(rf"\b{re.escape(n)}\b", txt) for n in nums):
                boosts[i] += 0.1

    return boosts

# -------- Rules JSON (Tabela) --------
def apply_table_rules(df: pd.DataFrame, row_texts: List[str], query: str, rules: dict):
    """
    Regras em JSON para filtrar/boostear linhas e expandir a consulta.
    Schema aceito (tudo opcional):
    {
      "numeric_filters": [{"column": "...", "op": ">=", "value": 20}],
      "must_contain_terms": ["conjunto"],
      "column_contains_any": [{"columns": ["Produto","Descrição"], "terms": ["conjunto","kit"]}],
      "regex_filters": ["\\bconjunto\\b"],
      "boosts": [{"columns": ["Produto","Descrição"], "contains": ["conjunto","kit"], "weight": 0.3}],
      "query_expand": {"20 nm": ["20 n.m","20nm","20 newton metro"]}
    }
    """
    import numpy as np
    import pandas as pd
    import re

    n = len(df)
    mask = np.ones(n, dtype=bool)
    extra_boosts = np.zeros(n, dtype=float)
    q_append = ""

    if not rules:
        return mask, extra_boosts, query

    # expand query
    if isinstance(rules.get("query_expand"), dict):
        q_low = (query or "").lower()
        extras = []
        for k, exp_list in rules["query_expand"].items():
            if str(k).lower() in q_low:
                extras.extend([str(x) for x in exp_list])
        if extras:
            q_append = " " + " ".join(extras)

    # numeric filters
    for f in rules.get("numeric_filters", []):
        col = f.get("column")
        op = f.get("op", ">=")
        val = f.get("value")
        if col is None or val is None or col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        cond = pd.Series(False, index=df.index)
        if op == ">=": cond = ser >= val
        elif op == ">": cond = ser > val
        elif op == "<=": cond = ser <= val
        elif op == "<": cond = ser < val
        elif op == "==": cond = ser == val
        elif op == "!=": cond = ser != val
        mask &= cond.fillna(False).to_numpy()

    # must contain all terms (row-text)
    terms = [t.lower() for t in rules.get("must_contain_terms", [])]
    if terms:
        merged = [str(t).lower() for t in row_texts]
        cond = np.array([all(term in merged[i] for term in terms) for i in range(n)], dtype=bool)
        mask &= cond

    # any term in specific columns
    for rule in rules.get("column_contains_any", []):
        cols = [c for c in rule.get("columns", []) if c in df.columns]
        want = [w.lower() for w in rule.get("terms", [])]
        if not cols or not want:
            continue
        colvals = (df[cols].fillna("").astype(str)).apply(lambda s: " | ".join(s), axis=1).str.lower().tolist()
        cond = np.array([any(w in colvals[i] for w in want) for i in range(n)], dtype=bool)
        mask &= cond

    # regex filters
    for patt in rules.get("regex_filters", []):
        try:
            rg = re.compile(patt, re.IGNORECASE)
        except re.error:
            continue
        cond = np.array([bool(rg.search(row_texts[i])) for i in range(n)], dtype=bool)
        mask &= cond

    # boosts
    for b in rules.get("boosts", []):
        weight = float(b.get("weight", 0.2))
        cols = [c for c in b.get("columns", []) if c in df.columns]
        terms_b = [t.lower() for t in b.get("contains", [])]
        if cols and terms_b:
            colvals = (df[cols].fillna("").astype(str)).apply(lambda s: " | ".join(s), axis=1).str.lower().tolist()
            hit = np.array([any(t in colvals[i] for t in terms_b) for i in range(n)], dtype=float)
            extra_boosts += weight * hit

    return mask, extra_boosts, (query or "") + q_append

# -------- EASY SERVO helpers --------
def excel_letter_to_index(letter: str) -> int:
    s = 0
    for ch in letter.strip().upper():
        if not ('A' <= ch <= 'Z'):
            continue
        s = s * 26 + (ord(ch) - 64)
    return s - 1  # 0-based

def excel_index_to_letter(idx: int) -> str:
    s, n = "", idx + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

YES_TOKENS = {"sim","x","✓","yes","true","1"}

def is_yes(value) -> bool:
    if value is None:
        return False
    txt = str(value).strip().lower()
    return txt in YES_TOKENS

def best_match_row_by_col(df, col_name: str, query: str) -> Optional[int]:
    if col_name not in df.columns:
        return None
    series = df[col_name].astype(str).fillna("")
    q = query.lower()
    nums = re.findall(r"\d+(?:[\.,]\d+)?", q)
    unit_hints = ["nm", "n.m", "n·m", "newton", "torque", "kit", "conjunto"]
    best_idx, best_score = None, -1
    for i, txt in series.items():
        t = txt.lower()
        score = SequenceMatcher(None, t, q).ratio()
        if any(n in t for n in nums):
            score += 0.05
        if any(u in t for u in unit_hints):
            score += 0.03
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

def apply_easy_servo_rule(df, query: str,
                          produtos_col_hint="produtos",
                          regra_col_hint="regra",
                          start_letter="D", end_letter="AC"):
    cols_l = {str(c).strip().lower(): c for c in df.columns}
    produtos_col = cols_l.get(produtos_col_hint.lower(), df.columns[0])  # fallback to first col
    regra_col = cols_l.get(regra_col_hint.lower(), df.columns[1] if len(df.columns) > 1 else df.columns[0])

    ridx = best_match_row_by_col(df, produtos_col, query)
    if ridx is None:
        return {"ok": False, "msg": f"Coluna '{produtos_col_hint}' não encontrada."}
    regra_val = df.at[ridx, regra_col] if regra_col in df.columns else ""

    start_idx = max(0, excel_letter_to_index(start_letter))
    end_idx = min(len(df.columns) - 1, excel_letter_to_index(end_letter))

    itens = []
    for cidx in range(start_idx, end_idx + 1):
        colname = df.columns[cidx]
        mark_cell = df.iloc[ridx, cidx]     # marcação "SIM" na linha do conjunto
        if is_yes(mark_cell):
            label_row2 = df.iloc[0, cidx]   # rótulo da linha 2 (Excel) → iloc[0] (pandas)
            itens.append({
                "excel_col": excel_index_to_letter(cidx),
                "col_name": str(colname),
                "label_row2": label_row2,
            })

    return {
        "ok": True,
        "linha_excel": int(ridx) + 2,  # pandas idx -> Excel linha
        "produto": df.at[ridx, produtos_col],
        "regra": regra_val,
        "itens": itens
    }

# -------- LLM prompt & call --------
SYSTEM_PROMPT = """Você é um assistente que responde ESTRITAMENTE com base no contexto de linhas/colunas de uma TABELA ou de um JSON.
- Se a informação não estiver claramente no contexto, responda: "Não encontrei essa informação nos dados fornecidos."
- Ao citar, inclua o nome da coluna e/ou o índice da linha quando possível.
- Seja objetivo e responda em português do Brasil.
"""

def build_llm_prompt_from_rows(question: str, df: pd.DataFrame, row_indices: List[int]) -> str:
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
        if len(row_txt) > 1800:
            row_txt = row_txt[:1800] + "…"
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

def call_hf_llm(model_id: str, prompt: str, token: str,
                max_new_tokens: int = 300, temperature: float = 0.2, timeout: int = 60):
    if not HF_AVAILABLE:
        return False, "Pacote huggingface_hub não está disponível."
    try:
        client = InferenceClient(model=model_id, token=token, timeout=timeout)
        try:
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
        except Exception as e1:
            err = str(e1).lower()
            if "not supported for provider" in err or "conversational" in err:
                try:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    resp = client.chat_completion(
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                    )
                    try:
                        content = resp.choices[0].message["content"]
                    except Exception:
                        try:
                            content = resp.choices[0].message.content
                        except Exception:
                            content = str(resp)
                    return True, content.strip()
                except Exception as e2:
                    return False, f"Falha no chat_completion: {e2}; erro original: {e1}"
            return False, f"Falha ao chamar o modelo Hugging Face: {e1}"
    except Exception as e:
        return False, f"Falha ao criar cliente HF: {e}"

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# ===================== UI =====================
st.title("🧠 NEI • Q&A para Tabelas e JSON (LLM + Regra EASY SERVO)")

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
    st.subheader("Regras")
    auto_easy = st.toggle("Aplicar automaticamente a regra EASY SERVO quando a aba corresponder", value=True)

# -------------------- Tabela (Excel/CSV) --------------------
if mode == "Tabela (Excel/CSV)":
    st.subheader("Carregar tabela")
    file = st.file_uploader("Envie um .xlsx, .xls ou .csv", type=["xlsx","xls","csv"])
    example_btn = st.button("Carregar exemplo de tabela")

    df = None
    active_sheet = None

    if example_btn:
        # Exemplo mínimo: rótulos na "linha 2" em D/I/P/R/V para ilustrar a regra
        df = pd.DataFrame({
            "produtos": ["Conjunto (KIT) 2Nm VDC 5m", "Conjunto Easy Servo 20 Nm", "Servo Panasonic 400W", "Motor NEMA 23"],
            "regra": ["1 Driver + 1 Motor + 1 Fonte + 1 Cabo Potência + 1 Cabo Encoder",
                      "1 Driver + 1 Motor + 2 Cabos", "Servo 400W sem freio", "Stepper"],
            "Torque (Nm)": [2, 20, np.nan, np.nan],
            "D": ["Driver", "", "", ""],
            "I": ["Motor",  "", "", ""],
            "P": ["Fonte",  "", "", ""],
            "R": ["Cabo Potência", "", "", ""],
            "V": ["Cabo Encoder", "", "", ""],
        })
        active_sheet = "EASY SERVO (PULSO E DIREÇÃO)"
    elif file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
            active_sheet = "(csv)"
        else:
            xls = pd.ExcelFile(file)
            default_sheet = "EASY SERVO (PULSO E DIREÇÃO)"
            if default_sheet in xls.sheet_names:
                sheet = st.selectbox("Planilha", xls.sheet_names, index=xls.sheet_names.index(default_sheet))
            else:
                sheet = st.selectbox("Planilha", xls.sheet_names, index=0)
            active_sheet = sheet
            df = pd.read_excel(xls, sheet_name=sheet)
    else:
        st.info("Envie sua planilha (.xlsx/.csv) ou clique em **Carregar exemplo de tabela**.")
        st.stop()

    if df is None or df.empty:
        st.error("Não foi possível carregar a tabela ou ela está vazia.")
        st.stop()

    st.success(f"Tabela carregada! {df.shape[0]} linhas × {df.shape[1]} colunas (aba: {active_sheet})")
    with st.expander("Ver dados"):
        st.dataframe(df, use_container_width=True)

    # Indexação para a busca geral
    row_texts = [make_row_text(row) for _, row in df.iterrows()]
    vec, mat = build_tfidf(row_texts)

    # Campo de Regras (JSON) para a busca geral
    with st.expander("⚙️ Regras de pesquisa (opcional)"):
        st.caption("Escreva regras em JSON para filtrar/boostear.")
        default_rules_json = '''{
  "numeric_filters": [{"column": "Torque (Nm)", "op": ">=", "value": 20}],
  "must_contain_terms": ["conjunto"],
  "boosts": [{"columns": ["produtos","regra"], "contains": ["conjunto","kit"], "weight": 0.4}],
  "query_expand": {"20 nm": ["20 n.m","20nm","20 newton metro"]}
}'''
        rules_text = st.text_area("Regras (JSON)", value=default_rules_json, height=220)
        rules = {}
        try:
            rules = json.loads(rules_text) if rules_text.strip() else {}
        except Exception as e:
            st.error(f"JSON de regras inválido: {e}")
            rules = {}

    # Pergunta do usuário
    q = st.chat_input("Pergunte algo… (ex.: 'Conjunto 2Nm')")

    # Nome da aba normalizado
    sheet_target = "EASY SERVO (PULSO E DIREÇÃO)"
    is_easy_servo_sheet = strip_accents((active_sheet or "").lower().strip()) == strip_accents(sheet_target.lower())

    if q:
        # 1) Aplica a regra EASY SERVO automaticamente quando a aba corresponder
        if auto_easy and is_easy_servo_sheet:
            res = apply_easy_servo_rule(df, q,
                                        produtos_col_hint="produtos",
                                        regra_col_hint="regra",
                                        start_letter="D", end_letter="AC")
            if not res.get("ok"):
                st.error(res.get("msg", "Falha na regra EASY SERVO."))
            else:
                st.markdown("### Resultado (Regra EASY SERVO)")
                st.success(f"Linha encontrada (Excel): **{res['linha_excel']}**")
                st.write("**Produto (A):**", res["produto"])
                st.write("**Regra (B):**", res["regra"])
                st.write("**Itens marcados (D..AC → valor da linha 2):**")
                if res["itens"]:
                    for it in res["itens"]:
                        st.write(f"- Coluna **{it['excel_col']}2** → `{it['label_row2']}` (nome: {it['col_name']})")
                else:
                    st.info("Nenhuma coluna D..AC marcada com 'SIM' nessa linha.")

                # Redigir com LLM (opcional)
                if use_llm and hf_token and res.get("itens"):
                    contexto = "\n".join([f"{it['excel_col']}2={it['label_row2']}" for it in res["itens"]])
                    prompt = f"""Você é um assistente. Com base no contexto abaixo, responda à pergunta.
Contexto:
- Produto: {res['produto']}
- Regra: {res['regra']}
- Itens: {contexto}

Pergunta: {q}
Responda objetivamente em PT-BR."""
                    ok, out = call_hf_llm(model_id, prompt, hf_token, max_new_tokens=220, temperature=0.2)
                    st.markdown("**Resposta (LLM):**" if ok else "**Erro LLM:**")
                    st.write(out)

                st.markdown("---")
                st.caption("Regra aplicada: procurar a linha na coluna A (produtos), ler B (regra) e, em D..AC, retornar o rótulo da LINHA 2 onde houver 'SIM'.")

        # 2) Busca geral (TF-IDF + regras JSON + boosts)
        mask_rules, rule_boosts, q_effective = apply_table_rules(df, row_texts, q, rules)
        if vec is None or mat is None:
            st.warning("Não consegui indexar a tabela.")
            st.stop()

        q_vec = vec.transform([q_effective])
        sims = cosine_similarity(q_vec, mat).ravel()
        boosts_smart = smart_boost_table(q_effective, df, row_texts)
        final_scores = sims + boosts_smart + rule_boosts
        final_scores[~mask_rules] = -1e9

        topk = np.argsort(-final_scores)[:6]
        st.markdown("### Linhas mais relevantes (busca geral)")
        subset_cols = list(df.columns)[: min(8, len(df.columns))]
        st.dataframe(df.iloc[topk][subset_cols], use_container_width=True)

        if use_llm and hf_token:
            prompt = build_llm_prompt_from_rows(q, df, list(topk))
            ok, out = call_hf_llm(model_id, prompt, hf_token, max_new_tokens=300, temperature=0.2)
            with st.chat_message("assistant"):
                st.markdown(out if ok else f"Não foi possível usar o LLM: {out}")

    # -------- Expander manual (opcional) --------
    with st.expander("🧩 Regra EASY SERVO (manual)"):
        st.caption("Execute manualmente a regra, independentemente da aba ativa.")
        usar_regra = st.toggle("Ativar regra EASY SERVO", value=False)
        pergunta = st.text_input("Pergunta (ex.: Conjunto 2Nm)", value="Conjunto 2Nm")
        faixa = st.text_input("Faixa de colunas (Excel)", value="D:AC")
        produtos_hint = st.text_input("Nome da coluna de produtos (A)", value="produtos")
        regra_hint = st.text_input("Nome da coluna de regra (B)", value="regra")
        if usar_regra and pergunta:
            try:
                ini, fim = [s.strip() for s in faixa.split(":")]
            except Exception:
                ini, fim = "D", "AC"
            res = apply_easy_servo_rule(df, pergunta,
                                        produtos_col_hint=produtos_hint,
                                        regra_col_hint=regra_hint,
                                        start_letter=ini, end_letter=fim)
            if not res.get("ok"):
                st.error(res.get("msg", "Falha na regra."))
            else:
                st.success(f"Linha encontrada (Excel): {res['linha_excel']}")
                st.write("**Produto (A):**", res["produto"])
                st.write("**Regra (B):**", res["regra"])
                st.write("**Itens marcados (D..AC → valor da linha 2):**")
                for it in res["itens"]:
                    st.write(f"- Coluna {it['excel_col']}2 → `{it['label_row2']}` (nome: {it['col_name']})")

# -------------------- JSON (compat) --------------------
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

    def flatten_json(obj: Any, path: str = "") -> List[dict]:
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

    q = st.chat_input("Pergunte algo sobre o JSON…")
    if q:
        idx, sims = cosine_topk(vec, mat, q, top_k=8)
        hits = df.iloc[idx] if idx.size else df.iloc[:0]
        if hits.empty:
            with st.chat_message("assistant"):
                st.markdown("Não encontrei nada diretamente relacionado no JSON.")
        else:
            lines = ["Aqui está o que encontrei relacionado:"]
            for _, row in hits.iterrows():
                val = row["value_str"]
                if len(val) <= 80 and "\n" not in val:
                    lines.append(f"- **{row['path']}** ({row['type']}): {val}")
                else:
                    lines.append(f"- **{row['path']}** ({row['type']}):\n```\n{val}\n```")
            with st.chat_message("assistant"):
                st.markdown("\n".join(lines))

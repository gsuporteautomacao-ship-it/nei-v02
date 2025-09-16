# NEI • Q&A para Tabelas e JSON (Streamlit + Hugging Face LLM)
- Busca TF-IDF + reforço por número/unidade
- Regras de pesquisa (JSON): filtros, boosts, expansão de consulta
- Regra EASY SERVO (PULSO E DIREÇÃO): A/B + D..AC com "SIM" → retorna rótulos da linha 2
- LLM com fallback (text_generation → chat_completion)

## Streamlit Cloud
1) Suba `app.py`, `requirements.txt`, `README.md` para um repositório.  
2) Conecte no Community Cloud e selecione `app.py`.  
3) (Opcional) Secret `HF_TOKEN`.  

## Dica de regras
```json
{
  "numeric_filters": [{"column": "Torque (Nm)", "op": ">=", "value": 20}],
  "must_contain_terms": ["conjunto"],
  "boosts": [{"columns": ["produtos","regra"], "contains": ["conjunto","kit"], "weight": 0.4}],
  "query_expand": {"20 nm": ["20 n.m","20nm","20 newton metro"]}
}

# JSON Q&A Bot (Streamlit + Hugging Face LLM)

App no Streamlit que responde perguntas sobre um **arquivo JSON**, com duas opções de resposta:
1) Heurística local (TF‑IDF)
2) **LLM via Hugging Face Inference API** (recomendado)

## Passos para publicar no streamlit.io

1. Crie um repositório no GitHub com estes arquivos: `app.py`, `requirements.txt`, `sample.json` e `README.md`.
2. No Streamlit Community Cloud, conecte o repositório e selecione o arquivo `app.py`.
3. **(Opcional LLM)** Adicione um **Secret** chamado `HF_TOKEN` com seu token do Hugging Face:
   - Vá em *Settings → Secrets*, cole:
     ```
     HF_TOKEN = "hf_xxx_seu_token_aqui"
     ```
4. Faça o deploy. No app, ative **"Usar LLM"** na barra lateral e escolha o `Model ID` (ex.: `meta-llama/Meta-Llama-3.1-8B-Instruct`).

> Dica: alguns modelos gratuitos podem estar ocupados/limitar tokens. Se ocorrer erro/timeout, troque de modelo ou reduza `max_new_tokens`.

## Como funciona

- O JSON é "achatado" em pares `path -> valor`.
- Uma busca TF‑IDF seleciona os trechos mais relevantes ao que foi perguntado.
- O contexto é enviado ao LLM com uma instrução para responder **apenas com base nesses dados**.
- Se o LLM estiver desativado ou sem token, a resposta heurística lista os caminhos e valores relevantes.

## Exemplos de modelos (Hugging Face)

- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `google/gemma-2-9b-it`

## Rodar localmente

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx  # ou configure no app
streamlit run app.py
```

## Segurança

- O JSON é processado apenas em memória do app.
- O LLM recebe apenas os trechos relevantes, não o arquivo completo.
- Não armazena o token; usa sessão/Secrets do Streamlit.
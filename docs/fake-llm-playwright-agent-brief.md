# Agent Brief: Fake LLM Replacement For RAGAS Dataset Pipeline

## Muc tieu

Tai lieu nay danh cho agent tiep theo doc nhanh de hieu he thong hien tai va tap trung vao bai toan thay LLM local/self-hosted bang mot fake LLM dieu khien ChatGPT GPT-5 qua Playwright.

Muc tieu ky thuat: giu pipeline tao RAG evaluation dataset hien co, nhung thay endpoint LLM dang tro ve `localhost`, Ollama, hoac OpenAI-compatible gateway bang mot adapter/proxy co the nhan request tu LangChain/RAGAS va lay cau tra loi tu ChatGPT UI bang Playwright.

## He thong dang lam gi

Day la mot Python CLI pipeline offline de tao dataset danh gia RAG tu tai lieu PDF va Markdown.

He thong khong phai web app va khong start backend server mac dinh.

Luon chay chinh:

1. CLI load bien moi truong tu `.env`.
2. Doc input folder chua `.pdf`, `.md`, hoac `.markdown`.
3. Trich xuat text tu PDF bang PyMuPDF hoac doc Markdown truc tiep.
4. Chia text thanh chunk bang LangChain text splitter.
5. Tao model wrapper cho RAGAS theo provider config.
6. Goi RAGAS `TestsetGenerator` de sinh cau hoi, context, va dap an tham chieu.
7. Normalize output ve schema cua du an.
8. Export ra `.jsonl` hoac `.csv`.

Output schema chinh:

```json
{
  "question": "Generated user question",
  "contexts": ["Reference source context"],
  "ground_truth": "Reference answer"
}
```

## Cong nghe va framework dang dung

Runtime va CLI:

- Python 3.10+
- Typer cho command line interface
- python-dotenv de load `.env`
- pandas de xu ly bang output
- pytest cho tests

Document processing:

- PyMuPDF (`pymupdf`) de doc PDF
- LangChain text splitters de chia chunk

LLM/RAG:

- RAGAS de generate evaluation testset
- LangChain wrappers de dua LLM/embedding vao RAGAS
- `langchain-openai` cho OpenAI va OpenAI-compatible provider
- `langchain-ollama` cho Ollama provider

## File quan trong

- `src/evaluation_dataset/cli.py`
  - Entry point CLI: `python -m src.evaluation_dataset.cli generate`
  - Load `.env`, tao config, doc documents, chunk, generate, export.

- `src/evaluation_dataset/config.py`
  - Dinh nghia `EvaluationDatasetConfig`.
  - Supported providers hien tai: `openai`, `openai-compatible`, `ollama`.
  - Doc config tu CLI, `.env`, sau do fallback source defaults.

- `src/evaluation_dataset/model_provider.py`
  - Diem chinh can sua neu muon thay LLM.
  - Tao `RagasModels(llm, embeddings)`.
  - Hien tai build 3 provider:
    - `_build_openai_models`
    - `_build_openai_compatible_models`
    - `_build_ollama_models`

- `src/evaluation_dataset/generator.py`
  - Goi `build_ragas_models(config)`.
  - Dua `models.llm` va `models.embeddings` vao RAGAS `TestsetGenerator`.
  - Dung cung LLM cho cac synthesizer:
    - `SingleHopSpecificQuerySynthesizer`
    - `MultiHopSpecificQuerySynthesizer`
    - `MultiHopAbstractQuerySynthesizer`

- `docs/rag-evaluation-dataset-usage-guide.md`
  - Huong dan su dung pipeline hien tai.

- `tests/evaluation_dataset/test_model_provider.py`
  - Test config/provider behavior.
  - Can update neu them provider moi.

## Config provider hien tai

Bien moi truong lien quan:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=http://localhost:11434
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:11434
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
RAGAS_TIMEOUT=600
RAGAS_NUM_PREDICT=2048
RAGAS_TEMPERATURE=0.0
RAGAS_LLM_FORMAT=json
```

OpenAI-compatible example:

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=fake-gpt-5
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:8000/v1
RAGAS_EMBEDDINGS_MODEL=fake-embeddings
OPENAI_API_KEY=not-needed
```

Important: RAGAS can require both an LLM and embeddings. A fake ChatGPT UI LLM only solves the generation side. Embeddings still need a real implementation or deterministic fake embeddings good enough for RAGAS generation.

## Contract can dap ung khi thay LLM

`generator.py` expects:

```python
RagasModels(
    llm=<RAGAS-compatible LLM wrapper>,
    embeddings=<RAGAS-compatible embeddings wrapper>,
)
```

RAGAS does not call an arbitrary function directly. It calls through LangChain/RAGAS wrapper interfaces.

Therefore there are two practical integration paths:

### Option A: OpenAI-compatible local proxy

Build a local HTTP server exposing enough OpenAI-compatible API for `langchain_openai.ChatOpenAI` and `OpenAIEmbeddings`.

Expected LLM endpoint:

```text
POST /v1/chat/completions
```

Minimum response shape:

```json
{
  "id": "chatcmpl-fake",
  "object": "chat.completion",
  "created": 0,
  "model": "fake-gpt-5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

Expected embeddings endpoint:

```text
POST /v1/embeddings
```

Minimum response shape:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0, 0.1, 0.2]
    }
  ],
  "model": "fake-embeddings"
}
```

Pros:

- Khong can sua nhieu code hien tai.
- Chi can set `RAGAS_PROVIDER=openai-compatible`.
- `model_provider.py` da support `base_url`.

Cons:

- Phai implement ca `/v1/chat/completions` va `/v1/embeddings`.
- ChatGPT UI qua Playwright cham, de rate limit, va khong on dinh cho batch generation.

### Option B: Them provider moi trong source code

Them provider vi du `playwright-chatgpt` vao:

- `SUPPORTED_PROVIDERS` trong `config.py`
- `_default_llm_model`, `_default_embeddings_model`, default temperature/format neu can
- `build_ragas_models` trong `model_provider.py`
- Tests trong `tests/evaluation_dataset/test_model_provider.py`

Provider moi se tao custom LangChain-compatible LLM class, sau do wrap bang `LangchainLLMWrapper`.

Pros:

- Kiem soat logic Playwright truc tiep trong Python.
- Co the custom retry, session, prompt formatting.

Cons:

- Can hieu ky interface LangChain LLM/chat model ma RAGAS chap nhan.
- De bi lech contract hon Option A.
- Van can embeddings.

Recommended: bat dau voi Option A vi code hien tai da co OpenAI-compatible path.

## Fake LLM bang ChatGPT GPT-5 qua Playwright

Y tuong proxy:

1. Local server nhan OpenAI-compatible `POST /v1/chat/completions`.
2. Server convert request messages thanh mot prompt duy nhat.
3. Playwright mo ChatGPT web UI da login san.
4. Gui prompt vao chat box.
5. Doi assistant response hoan tat.
6. Lay text response.
7. Tra ve OpenAI-compatible JSON cho LangChain.

Can luu y:

- RAGAS thuong yeu cau output co cau truc JSON cho mot so buoc extraction/synthesis.
- Neu dung fake UI, prompt phai yeu cau ChatGPT chi tra ve JSON khi request goc can JSON.
- Nen dat temperature cua pipeline la `0.0` neu endpoint fake co tham so nay.
- Can serialize requests. ChatGPT UI khong nen nhan nhieu prompt song song trong cung mot browser session.
- Can timeout dai. RAGAS co nhieu call lien tiep.
- Can retry khi UI bi mat focus, response chua xong, captcha, login expired, hoac UI thay doi selector.

## Embeddings la van de rieng

ChatGPT UI khong cung cap embeddings.

Cac lua chon:

1. Dung real embeddings rieng:
   - OpenAI embeddings
   - Ollama `nomic-embed-text`
   - sentence-transformers local endpoint

2. Fake embeddings endpoint:
   - Tao deterministic vector tu hash cua text.
   - Phu hop smoke test pipeline, nhung chat luong RAGAS co the kem.
   - Vector dimension phai co dinh cho moi request.

3. Tach LLM va embeddings base URL:
   - Code hien tai cho phep `RAGAS_LLM_BASE_URL` va `RAGAS_EMBEDDINGS_BASE_URL` khac nhau.
   - Co the fake LLM bang Playwright, nhung embeddings dung Ollama/local model.

Recommended for practical run:

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=fake-gpt-5
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:11434/v1
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
OPENAI_API_KEY=not-needed
```

Neu Ollama embeddings khong expose OpenAI-compatible `/v1/embeddings`, can dung provider rieng hoac tao proxy embeddings.

## Noi can thay doi toi thieu neu dung Option A

Co the khong can sua source code pipeline.

Chi can:

1. Build local OpenAI-compatible proxy.
2. Set `.env`:

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=fake-gpt-5
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:8000/v1
RAGAS_EMBEDDINGS_MODEL=fake-embeddings
OPENAI_API_KEY=not-needed
RAGAS_TIMEOUT=600
RAGAS_TEMPERATURE=0.0
```

3. Run smoke test:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path docs/KB/BachDang/rag_eval_dataset.fake-llm.sample.jsonl `
  --output-format jsonl `
  --testset-size 1
```

## Noi can thay doi neu dung Option B

`src/evaluation_dataset/config.py`:

```python
SUPPORTED_PROVIDERS = {"openai", "openai-compatible", "ollama", "playwright-chatgpt"}
```

`src/evaluation_dataset/model_provider.py`:

```python
def build_ragas_models(config: EvaluationDatasetConfig) -> RagasModels:
    settings = build_model_settings(config)
    if settings.provider == "playwright-chatgpt":
        return _build_playwright_chatgpt_models(settings)
```

Sau do implement:

```python
def _build_playwright_chatgpt_models(settings: ModelSettings) -> RagasModels:
    ...
```

Can dam bao returned objects duoc wrap dung:

```python
RagasModels(
    llm=LangchainLLMWrapper(custom_langchain_llm),
    embeddings=LangchainEmbeddingsWrapper(custom_embeddings),
)
```

## RAGAS parser risk

RAGAS rat nhay voi output khong dung format. Loi thuong gap:

```text
RagasOutputParserException
JSONDecodeError
Invalid \uXXXX escape
```

Khi fake LLM qua ChatGPT UI, can uu tien:

- Prompt system yeu cau output dung schema, khong markdown fence neu RAGAS can raw JSON.
- Strip markdown code fence truoc khi tra response neu can.
- Preserve UTF-8 tieng Viet, khong tu escape Unicode sai.
- Log raw prompt va raw response de debug.
- Bat dau voi `--testset-size 1`.

## Validation sau khi implement

Chay tests hien co:

```powershell
python -m pytest tests
```

Kiem tra CLI:

```powershell
python -m src.evaluation_dataset.cli --help
```

Smoke test voi fake LLM:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path docs/KB/BachDang/rag_eval_dataset.fake-llm.sample.jsonl `
  --output-format jsonl `
  --testset-size 1
```

Kiem tra output file co it nhat cac field:

- `question`
- `contexts`
- `ground_truth`

## Khuyen nghi cho agent tiep theo

Nen lam theo thu tu:

1. Giu source pipeline nguyen trang ban dau.
2. Build proxy OpenAI-compatible o ngoai hoac trong module rieng.
3. Fake `/v1/chat/completions` bang Playwright ChatGPT UI.
4. Implement `/v1/embeddings` bang deterministic hash vector hoac proxy sang embeddings that.
5. Chay `--testset-size 1`.
6. Neu RAGAS parse fail, log raw response va sua response cleanup trong proxy.
7. Chi them provider moi vao source code neu OpenAI-compatible proxy khong dap ung duoc.

Trong bai toan nay, diem kho khong nam o CLI pipeline. Diem kho nam o viec fake service phai thoa man dung API contract ma `langchain_openai` va RAGAS mong doi.

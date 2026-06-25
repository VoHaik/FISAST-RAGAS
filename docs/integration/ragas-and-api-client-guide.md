# Fake LLM Proxy Integration Guide

Guide này dành cho source khác muốn gọi Fake LLM Playwright Proxy như một OpenAI-compatible API. Trường hợp chính là source RAGAS dataset pipeline đã mô tả trong `fake-llm-playwright-agent-brief.md`.

## Tổng quan

Proxy chạy local và expose API:

```text
Base URL: http://localhost:8000/v1
Simple:   POST http://localhost:8000/v1/chat
Chat:     POST http://localhost:8000/v1/chat/completions
Embed:    POST http://localhost:8000/v1/embeddings
Sessions: GET  http://localhost:8000/v1/sessions
Scale:    POST http://localhost:8000/v1/sessions/scale
Health:   GET  http://localhost:8000/health
```

Luồng xử lý:

1. Source bên ngoài gửi request OpenAI-compatible vào proxy.
2. Proxy phân phối request chat vào pool ChatGPT tabs; nếu hết tab rảnh thì request sẽ chờ.
3. Playwright gõ prompt vào ChatGPT web UI đã login.
4. Proxy lấy assistant response, strip markdown fence nếu cần, rồi trả về JSON theo shape OpenAI-compatible.
5. Embeddings endpoint trả deterministic fake vector để smoke test. Dùng real embeddings nếu cần chất lượng RAGAS thật.

## Chạy proxy

Tạo và activate môi trường ảo local:

```powershell
C:\Users\KHAI\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn activate script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Sau khi activate, `python` sẽ trỏ vào `.\.venv\Scripts\python.exe`.

Install dependencies trong venv:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m playwright install chromium
```

Login ChatGPT lần đầu:

```powershell
$env:FAKE_LLM_HEADLESS="false"
python -m fake_llm_proxy
```

Khi browser mở ra, login ChatGPT thủ công. Session được lưu trong `.playwright-chatgpt-profile`.

Nếu Google sign-in báo "This browser or app may not be secure", giữ cấu hình này trong `.env` để Playwright mở Chrome đã cài trên máy thay vì Chromium bundled:

```env
FAKE_LLM_BROWSER_CHANNEL=chrome
```

Sau đó restart proxy và login lại trong cửa sổ Playwright. Nếu máy không có Chrome, thử:

```env
FAKE_LLM_BROWSER_CHANNEL=msedge
```

Nếu Google vẫn chặn, dùng CDP mode để proxy attach vào Chrome thật do bạn tự mở:

```powershell
& "$env:ProgramFiles\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$PWD\.chrome-cdp-profile" `
  https://chatgpt.com/
```

Login ChatGPT trong cửa sổ Chrome đó, rồi set trong `.env`:

```env
FAKE_LLM_CDP_URL=http://127.0.0.1:9222
```

Restart proxy. Khi `FAKE_LLM_CDP_URL` có giá trị, proxy sẽ connect vào Chrome đang mở thay vì tự launch browser mới.

Các lần sau:

```powershell
python -m fake_llm_proxy
```

Health check:

```powershell
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

## Dynamic sessions

Mặc định proxy start một ChatGPT tab:

```env
FAKE_LLM_SESSION_COUNT=1
```

Nếu muốn RAGAS hoặc source ngoài gọi nhiều request song song hơn, có thể tăng số tab lúc start bằng `.env`:

```env
FAKE_LLM_SESSION_COUNT=2
```

Hoặc scale runtime không cần restart server:

```powershell
curl.exe -X POST http://localhost:8000/v1/sessions/scale `
  -H "Content-Type: application/json" `
  -d "{\"count\":3}"
```

Kiểm tra trạng thái pool:

```powershell
curl http://localhost:8000/v1/sessions
```

Response:

```json
{"count":3,"available":3,"busy":0}
```

`count` là số tab ChatGPT đang được proxy quản lý, `available` là tab rảnh, `busy` là tab đang xử lý request. Scale down chỉ đóng tab idle; request đang chạy sẽ không bị kill giữa chừng. Nên bắt đầu với `2` hoặc `3` sessions vì ChatGPT web có thể chậm, đổi selector, hoặc rate limit nếu mở quá nhiều tab.

## API contract cho source khác

### Simple chat text API

Dùng endpoint này khi source ngoài chỉ cần gửi một text prompt và nhận text trả về.

Request:

```http
POST /v1/chat
Content-Type: application/json
```

Body:

```json
{
  "text": "Reply with exactly: SIMPLE_CHAT_OK"
}
```

PowerShell smoke request:

```powershell
curl.exe -X POST http://localhost:8000/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"Reply with exactly: SIMPLE_CHAT_OK\"}"
```

Response:

```json
{
  "text": "SIMPLE_CHAT_OK"
}
```

Lưu ý:

- Output không bị ép JSON. ChatGPT trả text, JSON string, Markdown, hoặc format khác thì proxy trả nguyên trong field `text`.
- Endpoint này phù hợp cho app custom muốn trigger ChatGPT trực tiếp.
- RAGAS không dùng endpoint này; RAGAS nên dùng OpenAI-compatible endpoint `/v1/chat/completions`.

### Chat completions

Đây là endpoint dành cho source dùng OpenAI-compatible client như `langchain-openai`, bao gồm source RAGAS.

Request:

```http
POST /v1/chat/completions
Content-Type: application/json
```

Body tối thiểu:

```json
{
  "model": "fake-gpt-5",
  "messages": [
    {
      "role": "user",
      "content": "Return raw JSON only: {\"ok\": true}"
    }
  ],
  "temperature": 0
}
```

PowerShell smoke request:

```powershell
curl.exe -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"fake-gpt-5\",\"messages\":[{\"role\":\"user\",\"content\":\"Return raw JSON only: {\\\"ok\\\": true}\"}],\"temperature\":0}"
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "fake-gpt-5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"ok\": true}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Lưu ý:

- `stream: true` không được support. Proxy sẽ trả HTTP 400.
- Request chat được phân phối vào session pool; mỗi ChatGPT tab chỉ nhận một prompt tại một thời điểm.
- Timeout nên đặt dài vì ChatGPT web chậm hơn API thật.

### Embeddings

Request:

```http
POST /v1/embeddings
Content-Type: application/json
```

Body:

```json
{
  "model": "fake-embeddings",
  "input": ["one", "two"]
}
```

PowerShell smoke request:

```powershell
curl.exe -X POST http://localhost:8000/v1/embeddings `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"fake-embeddings\",\"input\":[\"one\",\"two\"]}"
```

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456]
    }
  ],
  "model": "fake-embeddings",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

`fake-embeddings` chỉ phù hợp smoke test. Với RAGAS quality evaluation thật, trỏ embeddings sang OpenAI/Ollama/sentence-transformers hoặc một endpoint OpenAI-compatible embeddings thật.

## Tích hợp source RAGAS dataset pipeline

Source RAGAS hiện tại đã có provider `openai-compatible`, nên không cần thêm provider mới nếu dùng proxy này.

RAGAS sẽ gọi:

```text
POST http://localhost:8000/v1/chat/completions
POST http://localhost:8000/v1/embeddings
```

Không cấu hình RAGAS gọi `/v1/chat`; endpoint đó chỉ dành cho source custom cần input text/output text đơn giản.

Set `.env` trong source RAGAS:

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

Nếu có embeddings thật, cấu hình thực tế nên tách LLM và embeddings:

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=fake-gpt-5
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:11434/v1
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
OPENAI_API_KEY=not-needed
RAGAS_TIMEOUT=600
RAGAS_TEMPERATURE=0.0
```

Chạy smoke test nhỏ nhất từ source RAGAS:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path docs/KB/BachDang/rag_eval_dataset.fake-llm.sample.jsonl `
  --output-format jsonl `
  --testset-size 1
```

Kiểm tra output:

```powershell
python -c "import json, pathlib; p=pathlib.Path('docs/KB/BachDang/rag_eval_dataset.fake-llm.sample.jsonl'); row=json.loads(p.read_text(encoding='utf-8').splitlines()[0]); assert {'question','contexts','ground_truth'} <= set(row); print(row)"
```

Expected output có tối thiểu:

```json
{
  "question": "...",
  "contexts": ["..."],
  "ground_truth": "..."
}
```

## Ví dụ Python client cho source khác

Nếu chỉ cần input text/output text, gọi simple endpoint:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat",
    json={"text": "Reply with exactly: SIMPLE_CHAT_OK"},
    timeout=600,
)
response.raise_for_status()
print(response.json()["text"])
```

Nếu cần OpenAI-compatible shape, gọi `/v1/chat/completions`:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "fake-gpt-5",
        "messages": [
            {"role": "user", "content": "Return raw JSON only: {\"ok\": true}"}
        ],
        "temperature": 0,
    },
    timeout=600,
)
response.raise_for_status()
content = response.json()["choices"][0]["message"]["content"]
print(content)
```

Nếu source dùng `langchain-openai`, cấu hình tương đương:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="fake-gpt-5",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    temperature=0,
    timeout=600,
)
```

## Troubleshooting

### Proxy không start được

Kiểm tra Python launcher, venv và dependencies:

```powershell
C:\Users\KHAI\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m playwright install chromium
```

Trong shell hiện tại của workspace này, global `python`, `py`, và `python3` từng không có trên PATH. Nếu gặp lỗi command not found, tạo venv bằng full path Python trước; sau khi activate thì dùng `python` trong venv.

### ChatGPT yêu cầu login

Chạy headed mode:

```powershell
$env:FAKE_LLM_HEADLESS="false"
python -m fake_llm_proxy
```

Login trong browser Playwright. Không xóa `.playwright-chatgpt-profile` nếu muốn giữ session.

Nếu login bằng Google bị chặn với lỗi "This browser or app may not be secure", set:

```env
FAKE_LLM_BROWSER_CHANNEL=chrome
```

Nếu Chrome không có trên máy, dùng:

```env
FAKE_LLM_BROWSER_CHANNEL=msedge
```

Sau đó restart proxy. Bạn vẫn login thủ công; app không cần và không lưu password Google.

Nếu vẫn bị chặn, dùng CDP mode:

```powershell
& "$env:ProgramFiles\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$PWD\.chrome-cdp-profile" `
  https://chatgpt.com/
```

Sau khi login thành công trong Chrome đó, set:

```env
FAKE_LLM_CDP_URL=http://127.0.0.1:9222
```

### RAGAS bị lỗi parse JSON

Các lỗi thường gặp:

```text
RagasOutputParserException
JSONDecodeError
Invalid \uXXXX escape
```

Cách xử lý:

- Bắt đầu với `--testset-size 1`.
- Giữ `RAGAS_TEMPERATURE=0.0`.
- Đảm bảo prompt yêu cầu raw JSON, không markdown fence.
- Proxy đã strip markdown fences phổ biến, nhưng ChatGPT vẫn có thể trả thêm giải thích nếu prompt upstream không đủ chặt.

### RAGAS chạy rất chậm

Đây là giới hạn của ChatGPT web UI qua Playwright. Proxy không gửi nhiều prompt cùng lúc vào cùng một tab; nếu muốn tăng throughput, tăng `FAKE_LLM_SESSION_COUNT` hoặc gọi `/v1/sessions/scale`. Đặt timeout dài:

```env
RAGAS_TIMEOUT=600
```

### Kết quả RAGAS kém chất lượng

Nếu dùng `fake-embeddings`, đây chỉ là smoke test đường ống. Dùng real embeddings để đánh giá chất lượng nghiêm túc.

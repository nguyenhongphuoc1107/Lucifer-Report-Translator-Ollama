# Lucifer Report VI -> EN

Remote architecture: Streamlit Community Cloud -> Cloudflare Tunnel -> FastAPI proxy -> local Ollama.

## 1. Local PC

Install Ollama, then:

```bash
ollama pull qwen3:8b
ollama run qwen3:8b
```

Install Python dependencies:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r api_requirements.txt
```

Set environment variables from `api.env.example` and launch:

```bash
uvicorn ollama_api:app --host 127.0.0.1 --port 8000
```

Open another terminal and test:

```bash
curl -H "X-API-Key: YOUR_SECRET" http://127.0.0.1:8000/health
```

## 2. Cloudflare Tunnel

For a quick development test:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare will print a `https://xxxx.trycloudflare.com` URL.

For a stable production URL, create a named Cloudflare Tunnel and map a hostname such as:

```text
ollama-api.example.com -> http://localhost:8000
```

Then keep `cloudflared` running as a service.

## 3. GitHub

Push these files:

```text
app.py
ollama_api.py
requirements.txt
api_requirements.txt
README.md
.gitignore
secrets.example.toml
api.env.example
```

Do NOT commit:

```text
.env
.streamlit/secrets.toml
```

## 4. Streamlit Cloud

Deploy `app.py` from GitHub.

In Streamlit Cloud App settings -> Secrets, paste:

```toml
OLLAMA_API_URL = "https://ollama-api.example.com"
OLLAMA_API_KEY = "YOUR_SECRET"
OLLAMA_MODEL = "qwen3:8b"
```

The app will automatically read these values.

## 5. Important

Your PC must be on and Ollama + FastAPI + Cloudflare Tunnel must be running whenever the Streamlit app needs to translate using Remote Ollama.

The public Streamlit app does not access your PC's `localhost`. It accesses the HTTPS Cloudflare hostname, which tunnels to FastAPI on your PC.

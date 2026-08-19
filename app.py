import io
import re
import time

import pandas as pd
import requests
import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Lucifer Report VI → EN",
    page_icon="🌐",
    layout="wide",
)

# ==========================================
# SECRETS / DEFAULTS
# ==========================================
def secret_or(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
        return str(value) if value is not None else default
    except Exception:
        return default

DEFAULTS = {
    "provider": "ollama_remote",
    "model": "qwen3:8b",
    "maxOutputTokens": 12000,
    "requestTimeout": 240,

    # Gemini / OpenAI are optional fallback providers.
    "geminiKey": "",
    "geminiKeyPool": "",
    "geminiModel": "gemini-2.5-flash",

    "openaiKey": "",
    "openaiKeyPool": "",
    "openaiModel": "gpt-5-mini",
    "openaiEndpoint": "https://api.openai.com/v1/responses",

    # Remote Ollama Proxy: Streamlit Cloud -> HTTPS API -> local Ollama.
    "ollamaApiUrl": secret_or("OLLAMA_API_URL", ""),
    "ollamaApiKey": secret_or("OLLAMA_API_KEY", ""),
    "ollamaModel": secret_or("OLLAMA_MODEL", "qwen3:8b"),

    "stylePreset": "customer",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "glossary_text" not in st.session_state:
    st.session_state.glossary_text = (
        "Nhận ngay = Get Now\n"
        "Khuyến mãi = Promotion\n"
        "Tài khoản nạp lần 2 = Account with second deposit\n"
        "LSGD = Transaction History\n"
        "LSC = Betting History"
    )

if "en_text_area" not in st.session_state:
    st.session_state.en_text_area = ""

# ==========================================
# PROMPT / CORE LOGIC
# ==========================================
TRANSLATION_CORE = """You are a senior QA/QC Technical Writer translating Vietnamese into professional English for a customer-facing software testing / verification report.

PRIMARY OBJECTIVE
Translate faithfully and naturally. The English must communicate the exact same facts, conditions, scope, and issue behavior as the Vietnamese source. Improve wording for professional QA communication, but never change the meaning.

NON-NEGOTIABLE RULES
1. Do NOT summarize. Do NOT merge separate scenarios, accounts, issues, bullets, or test cases.
2. Preserve the source structure: headings, numbering, bullets, indentation, paragraphs, and order.
3. Preserve technical tokens exactly when they are identifiers or product UI: URLs, domains, usernames/account IDs, package names, version numbers, API names, error messages inside quotes, button labels inside quotes, field labels inside quotes, and acronyms.
4. Do not invent a requirement, expected result, root cause, environment, evidence, severity, priority, frequency, or technical explanation that is not stated in the source.
5. Do not soften or exaggerate the issue. Keep the same certainty level as the source.
6. Use QA/QC terminology that sounds natural to an experienced software tester, not literal machine translation.
7. When the source says “team verified/check”, use QA language such as “The team verified...” or “The team completed verification of...”, whichever is more natural.
8. “Hoạt động bình thường” in a verified-function context should normally become “works as expected” or “operates as expected”, not “operates normally”.
9. “Nhập chuỗi khoảng trắng” means “entering a whitespace-only string” or “entering a string containing only spaces”, depending on context.
10. “Hiển thị sai field” means “is displayed under the wrong field” / “appears on the wrong field”.
11. “Không hiển thị validation” means “no validation message is displayed/shown”.
12. “Btn” means “button”. “Field” means “field”. “Popup” may remain “popup” in QA/customer communication.
13. Prefer precise QA nouns: “deposit/withdrawal request” over “deposit/withdrawal ticket” unless the source clearly refers to a ticketing object; “validation message”; “Confirm Password field”; “Forgot Password link”; “Transaction History”; “Betting History”; “floating button”; “bottom navigation bar”.
14. Preserve product-specific terms such as “Promotion Account” and “Rebate Account” when they are account/product names. Do not casually rename them to generic marketing terms.
15. If a Vietnamese phrase has an acronym or local UI term (e.g. LSGD, LSC), keep the acronym and add a concise English descriptor only when the source context makes that meaning clear.
16. Do not turn the content into a formal Jira bug template unless the source already uses one. This tool is a translator, not a bug rewriter.
17. Keep customer-facing English concise, direct, and professional. Avoid overly conversational wording such as “the team sends you”. Prefer “The team is sharing the verification results below.”
18. Do not add greetings/sign-offs beyond what is present in the source.
19. Keep quotes and punctuation consistent with the source where practical.
20. Output ONLY the English translation. No commentary, no analysis, no notes about translation.

PREFERRED QA TERMINOLOGY EXAMPLES
- “Đăng ký/Đăng nhập/Đăng xuất” → “Register/Login/Logout”
- “Cập nhật thông tin tài khoản/ngân hàng/mật khẩu” → “Update account/bank/password information”
- “popup chặn khuyến mãi” → “promotion restriction popup” / “promotion-blocking popup” depending on context; prefer the clearest customer-facing term.
- “tạo phiếu nạp/rút” → “create a deposit/withdrawal request”
- “LSGD” → “Transaction History (LSGD)” when the acronym is important
- “LSC” → “Betting History (LSC)” when the acronym is important
- “tổng cược hôm nay” → “today’s total bet amount” or “today’s total betting amount” depending on surrounding context
- “số tiền cược” → “bet amount” / “betting amount”
- “không được cập nhật” → “is not updated” / “does not update”
- “dẫn về domain cũ” → “redirects to the old domain”
- “không disable” → “does not become disabled” / “remains enabled” depending on context
- “back từ popup” → “after returning from the popup”
- “thay đổi mật khẩu thành công” → “the password is successfully changed”
- “overlap” → “overlaps with” when describing UI collision

SOURCE SHOULD BE TREATED AS GROUND TRUTH. When wording is ambiguous, choose the most neutral QA phrasing that preserves the original uncertainty instead of inventing detail."""


def style_instructions(preset: str) -> str:
    if preset == "literal":
        return (
            "STYLE PRESET: STRICT / CLOSE-TO-SOURCE\n"
            "Stay very close to the original sentence order and wording, while fixing obvious English grammar and using correct QA terminology."
        )
    if preset == "internal":
        return (
            "STYLE PRESET: INTERNAL QA/QC\n"
            "Use concise internal QA language. Keep technical shorthand/acronyms where useful. Do not add customer-friendly explanation."
        )
    return (
        "STYLE PRESET: CUSTOMER-FACING QA/QC\n"
        "Write polished professional English suitable for sending directly to an external customer. Keep it factual and easy to follow without diluting technical detail. Replace awkward Vietnamese report phrasing with natural QA communication. For example, replace “The team sends you the check results” with “The team is sharing the verification results below.”"
    )


def build_prompt(source: str, project: str, domain: str, glossary: str, preset: str) -> str:
    prompt = style_instructions(preset) + "\n\n"
    if project:
        prompt += f"PROJECT / PRODUCT CONTEXT: {project}\n"
    if domain:
        prompt += f"DOMAIN / PRODUCT TERMINOLOGY CONTEXT: {domain}\n"
    if glossary:
        prompt += (
            "MANDATORY GLOSSARY (follow exactly unless doing so would contradict the source):\n"
            f"{glossary}\n\n"
        )

    prompt += (
        "TASK\n"
        "Translate the following Vietnamese report into English.\n"
        "- Preserve every distinct verification result and every distinct old issue.\n"
        "- Do not merge separate bullets merely because they discuss the same feature.\n"
        "- Preserve account distinctions such as new account vs. second-deposit account.\n"
        "- Preserve all quoted UI labels and values.\n"
        "- Preserve the final “=> details in file/link” line and translate only its surrounding wording.\n"
        "- If [Link] appears, keep [Link] exactly.\n\n"
    )
    return prompt + TRANSLATION_CORE + "\n\nVIETNAMESE SOURCE\n" + source


def split_into_chunks(text: str, max_chars: int = 14000):
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sections = re.split(
        r"\n(?=\d+\.\s|=>\s|[A-ZÀ-Ỵ][^\n]{0,80}:\s*$)",
        normalized,
    )

    chunks = []
    current = ""
    for part in sections:
        candidate = f"{current}\n{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(part) <= max_chars:
            current = part
            continue

        paragraphs = part.split("\n\n")
        sub = ""
        for p in paragraphs:
            candidate_p = f"{sub}\n\n{p}" if sub else p
            if len(candidate_p) <= max_chars:
                sub = candidate_p
            else:
                if sub:
                    chunks.append(sub)
                sub = p
        current = sub

    if current:
        chunks.append(current)

    final_chunks = []
    for chunk in chunks:
        for start in range(0, len(chunk), max_chars):
            final_chunks.append(chunk[start:start + max_chars])
    return final_chunks


def clean_model_output(text: str) -> str:
    if text is None:
        raise Exception("AI không trả về nội dung.")
    text = str(text).strip()
    text = re.sub(r"^```(?:text|markdown|en)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_keys(primary: str, pool: str):
    keys = []
    if primary and primary.strip():
        keys.append(primary.strip())
    keys.extend(line.strip() for line in (pool or "").splitlines() if line.strip())
    return list(dict.fromkeys(keys))

# ==========================================
# REMOTE OLLAMA PROXY
# ==========================================
def call_remote_ollama(system: str, user_text: str) -> str:
    base = str(st.session_state.ollamaApiUrl).strip().rstrip("/")
    api_key = str(st.session_state.ollamaApiKey).strip()
    model = str(st.session_state.ollamaModel).strip()

    if not base:
        raise Exception(
            "Chưa cấu hình OLLAMA_API_URL. Hãy nhập URL API trung gian, "
            "ví dụ https://ollama-api.example.com"
        )
    if not api_key:
        raise Exception("Chưa cấu hình OLLAMA_API_KEY.")
    if not model:
        raise Exception("Chưa cấu hình Ollama model.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.1,
        "max_output_tokens": int(st.session_state.maxOutputTokens),
    }

    response = requests.post(
        f"{base}/v1/chat",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        json=payload,
        timeout=int(st.session_state.requestTimeout),
    )

    if not response.ok:
        raise Exception(
            f"Remote Ollama API Error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()
    try:
        return clean_model_output(data["message"]["content"])
    except (KeyError, TypeError):
        raise Exception("Remote Ollama API không trả về message.content hợp lệ.")


def test_remote_ollama() -> str:
    base = str(st.session_state.ollamaApiUrl).strip().rstrip("/")
    api_key = str(st.session_state.ollamaApiKey).strip()
    if not base or not api_key:
        raise Exception("Thiếu Remote API URL hoặc API key.")

    response = requests.get(
        f"{base}/health",
        headers={"X-API-Key": api_key},
        timeout=min(30, int(st.session_state.requestTimeout)),
    )
    if not response.ok:
        raise Exception(
            f"Health check Error {response.status_code}: {response.text[:500]}"
        )
    return response.text

# ==========================================
# GEMINI / OPENAI OPTIONAL FALLBACKS
# ==========================================
def call_gemini(key, model, system, user_text, retry=0):
    max_tokens = min(30000, max(2000, int(st.session_state.maxOutputTokens)))
    response_tokens = min(max_tokens, 7000 if retry >= 2 else 9000 if retry >= 1 else max_tokens)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system}\n\nSOURCE TEXT:\n{user_text}"}]}],
        "generationConfig": {"maxOutputTokens": response_tokens},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        json=payload,
        timeout=int(st.session_state.requestTimeout),
    )
    if not response.ok:
        raise Exception(f"Gemini API Error {response.status_code}: {response.text[:500]}")

    data = response.json()
    try:
        return clean_model_output(data["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError):
        raise Exception("Gemini không trả về nội dung hợp lệ.")


def call_openai(system, user_text):
    keys = normalize_keys(st.session_state.openaiKey, st.session_state.openaiKeyPool)
    if not keys:
        raise Exception("Chưa nhập OpenAI API Key.")

    endpoint = str(st.session_state.openaiEndpoint).strip().rstrip("/")
    model = st.session_state.openaiModel
    last_err = None

    for key in keys:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }

            if endpoint.endswith("/responses"):
                payload = {
                    "model": model,
                    "input": [
                        {"role": "system", "content": [{"type": "input_text", "text": system}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                    ],
                    "max_output_tokens": int(st.session_state.maxOutputTokens),
                }
                response = requests.post(endpoint, headers=headers, json=payload, timeout=int(st.session_state.requestTimeout))
                if not response.ok:
                    raise Exception(f"OpenAI Responses API Error {response.status_code}: {response.text[:500]}")
                data = response.json()
                if data.get("output_text"):
                    return clean_model_output(data["output_text"])
                parts = []
                for item in data.get("output", []):
                    for content in item.get("content", []):
                        if content.get("text"):
                            parts.append(content["text"])
                if parts:
                    return clean_model_output("\n".join(parts))
                raise Exception("OpenAI không trả về output_text hợp lệ.")

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": int(st.session_state.maxOutputTokens),
                "temperature": 0.1,
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=int(st.session_state.requestTimeout))
            if not response.ok:
                raise Exception(f"OpenAI Chat API Error {response.status_code}: {response.text[:500]}")
            data = response.json()
            return clean_model_output(data["choices"][0]["message"]["content"])
        except Exception as exc:
            last_err = exc
            time.sleep(0.5)

    raise last_err or Exception("OpenAI request failed.")


def call_ai(system: str, user_text: str) -> str:
    provider = st.session_state.provider

    if provider == "ollama_remote":
        return call_remote_ollama(system, user_text)

    if provider == "gemini":
        keys = normalize_keys(st.session_state.geminiKey, st.session_state.geminiKeyPool)
        if not keys:
            raise Exception("Chưa nhập Gemini API Key.")
        last_err = None
        for key in keys:
            for retry in range(3):
                try:
                    return call_gemini(key, st.session_state.geminiModel, system, user_text, retry)
                except Exception as exc:
                    last_err = exc
                    time.sleep(0.5)
        raise last_err or Exception("Gemini request failed.")

    if provider == "openai":
        return call_openai(system, user_text)

    raise Exception(f"Provider không hợp lệ: {provider}")

# ==========================================
# UI
# ==========================================
st.title("LUCIFER REPORT VI → EN 🌐")
st.markdown("Customer-facing QA/QC • faithful translation • remote Ollama")

tab_translate, tab_settings, tab_guide = st.tabs(["🌐 VI → EN", "⚙️ Cài đặt", "📘 Hướng dẫn"])

with tab_translate:
    st.markdown("### Translation mode")
    st.selectbox(
        "Style preset",
        ["customer", "internal", "literal"],
        key="stylePreset",
        format_func=lambda x: {
            "customer": "Customer-facing QA/QC",
            "internal": "Internal QA/QC",
            "literal": "Strict / Close-to-source",
        }[x],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇻🇳 Nội dung tiếng Việt")
        vi_input = st.text_area("Dán nguyên report/check result/issue list...", height=400)
        st.caption(f"{len(vi_input)} chars")

        ctx_col1, ctx_col2 = st.columns(2)
        with ctx_col1:
            project_context = st.text_input("QA context / Project", placeholder="Tambet / Production")
        with ctx_col2:
            domain_context = st.text_input("Product domain", placeholder="sportsbook, promotion, deposit/withdrawal...")

        st.markdown("**Glossary / thuật ngữ bắt buộc**")
        glossary_file = st.file_uploader("📥 Import .txt / .csv", type=["txt", "csv"])

        if glossary_file is not None:
            try:
                content = glossary_file.getvalue().decode("utf-8-sig")
                new_terms = []
                if glossary_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(io.StringIO(content))
                    if df.shape[1] < 2:
                        raise ValueError("CSV phải có ít nhất 2 cột: VI và EN.")
                    for _, row in df.iterrows():
                        if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                            new_terms.append(f"{str(row.iloc[0]).strip()} = {str(row.iloc[1]).strip()}")
                else:
                    for line in content.splitlines():
                        line = line.strip()
                        if line and "=" in line and not line.startswith("#"):
                            new_terms.append(line)

                current_terms = st.session_state.glossary_text.splitlines()
                all_terms = list(dict.fromkeys(current_terms + new_terms))
                st.session_state.glossary_text = "\n".join(t for t in all_terms if t.strip())
                st.success(f"Import thành công: {len(new_terms)} thuật ngữ.")
            except Exception as exc:
                st.error(f"Lỗi import glossary: {exc}")

        glossary_text = st.text_area("Glossary (VI = EN)", value=st.session_state.glossary_text, height=150)
        st.session_state.glossary_text = glossary_text

        if st.button("🌐 Dịch sang English QA", type="primary", use_container_width=True):
            if not vi_input.strip():
                st.warning("Hãy nhập nội dung tiếng Việt.")
            else:
                with st.spinner(f"Đang dịch bằng {st.session_state.provider}..."):
                    try:
                        chunks = split_into_chunks(vi_input)
                        outputs = []
                        for idx, chunk in enumerate(chunks):
                            if len(chunks) > 1:
                                st.toast(f"Đang dịch phần {idx + 1}/{len(chunks)}...")

                            prompt = build_prompt(
                                chunk,
                                project_context,
                                domain_context,
                                glossary_text,
                                st.session_state.stylePreset,
                            )
                            system_instruction = TRANSLATION_CORE + "\n\n" + style_instructions(st.session_state.stylePreset)
                            out = call_ai(system_instruction, prompt.replace(f"{TRANSLATION_CORE}\n\n", ""))
                            outputs.append(clean_model_output(out))

                        st.session_state.en_text_area = "\n\n".join(outputs)
                        st.success("✅ Dịch hoàn tất!")
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")

    with col2:
        st.subheader("🇬🇧 English — Customer-facing QA/QC")
        st.text_area("Bản dịch hiển thị tại đây...", height=650, key="en_text_area")
        st.caption(f"{len(st.session_state.en_text_area)} chars")
        if st.session_state.en_text_area:
            st.download_button(
                "⬇️ Tải file TXT",
                data=st.session_state.en_text_area,
                file_name="Lucifer_Report_VI_EN.txt",
                mime="text/plain",
            )

with tab_settings:
    st.subheader("⚙️ AI Provider")
    st.caption("Khuyến nghị: Remote Ollama. Streamlit Cloud chỉ giữ giao diện; model chạy trên PC của bạn.")

    provider_options = ["ollama_remote", "openai", "gemini"]
    st.session_state.provider = st.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(st.session_state.provider),
        format_func=lambda x: {
            "ollama_remote": "🦙 Ollama Remote (FREE / Local PC)",
            "openai": "🤖 OpenAI API (Paid)",
            "gemini": "💎 Gemini API",
        }[x],
    )

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.maxOutputTokens = st.number_input(
            "Max output tokens", min_value=2000, max_value=30000, step=500,
            value=int(st.session_state.maxOutputTokens),
        )
    with c2:
        st.session_state.requestTimeout = st.number_input(
            "Request timeout (seconds)", min_value=30, max_value=900, step=30,
            value=int(st.session_state.requestTimeout),
        )

    if st.session_state.provider == "ollama_remote":
        st.markdown("#### 🦙 Remote Ollama")
        st.session_state.ollamaApiUrl = st.text_input(
            "Remote API URL",
            value=st.session_state.ollamaApiUrl,
            placeholder="https://ollama-api.yourdomain.com",
            help="URL của FastAPI proxy đang chạy trên PC/server và được publish bằng Cloudflare Tunnel.",
        )
        st.session_state.ollamaApiKey = st.text_input(
            "Remote API Key",
            value=st.session_state.ollamaApiKey,
            type="password",
            help="Nên lưu trong Streamlit Secrets thay vì commit vào GitHub.",
        )
        st.session_state.ollamaModel = st.text_input(
            "Ollama Model",
            value=st.session_state.ollamaModel,
        )
        st.info("PC của bạn chạy FastAPI + Ollama. Streamlit Cloud gửi HTTPS request tới FastAPI, không truy cập localhost trực tiếp.")

    elif st.session_state.provider == "openai":
        st.markdown("#### 🤖 OpenAI")
        st.session_state.openaiModel = st.text_input("OpenAI Model", value=st.session_state.openaiModel)
        st.session_state.openaiKey = st.text_input("OpenAI API Key chính", value=st.session_state.openaiKey, type="password")
        st.session_state.openaiKeyPool = st.text_area("OpenAI Key Pool (mỗi dòng 1 key)", value=st.session_state.openaiKeyPool)
        st.session_state.openaiEndpoint = st.text_input("OpenAI Endpoint", value=st.session_state.openaiEndpoint)

    else:
        st.markdown("#### 💎 Gemini")
        st.session_state.geminiModel = st.text_input("Gemini Model", value=st.session_state.geminiModel)
        st.session_state.geminiKey = st.text_input("Gemini API Key chính", value=st.session_state.geminiKey, type="password")
        st.session_state.geminiKeyPool = st.text_area("Gemini Key Pool (mỗi dòng 1 key)", value=st.session_state.geminiKeyPool)

    if st.button("🔌 Test AI", use_container_width=True):
        with st.spinner("Đang kiểm tra kết nối..."):
            try:
                if st.session_state.provider == "ollama_remote":
                    st.success("✅ Remote API OK: " + test_remote_ollama())
                else:
                    test_msg = "You are a QA translation API health check. Reply exactly: QA_TRANSLATION_OK"
                    out = call_ai(test_msg, test_msg)
                    if "QA_TRANSLATION_OK" in out:
                        st.success("✅ AI hoạt động.")
                    else:
                        st.error(f"Phản hồi không mong đợi: {out}")
            except Exception as exc:
                st.error(f"Lỗi: {exc}")

with tab_guide:
    st.markdown(
        """
### 📘 Kiến trúc mới

```text
Browser
   │
   ▼
Streamlit Community Cloud
   │ HTTPS + X-API-Key
   ▼
Cloudflare Tunnel
   │
   ▼
FastAPI Proxy trên PC
   │ HTTP localhost
   ▼
Ollama
   │
   ▼
Qwen3:8b
```

### Vì sao cách này giải quyết lỗi quota?

Streamlit không gọi Gemini để dịch mặc định nữa. Nội dung được gửi tới API trung gian của chính bạn, sau đó API trung gian gọi Ollama đang chạy trên PC.

### Bảo mật

Không commit API key vào GitHub. Với Streamlit Cloud, đặt `OLLAMA_API_URL`, `OLLAMA_API_KEY` và `OLLAMA_MODEL` trong Secrets.
"""
    )

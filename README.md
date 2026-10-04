# 🧠 Subtitle AI Translator (Kodi Add-on)

**Subtitle AI Translator** is a smart, user-friendly **Kodi add-on** that enables you to **translate subtitle files using Large Language Models (LLMs)** — currently supporting **OpenAI's GPT models** and **Google's Gemini API**.

It’s especially useful for users who want to enjoy movies and shows with subtitles in their preferred language, with preserved formatting and natural, fluent translations.

> ⚠️ This add-on is **experimental** and provided **as-is**. Use at your own risk.

---

## 🌟 Features

- 🔤 Translate `.srt` subtitle files (other formats planned)
- 🤖 Uses **OpenAI** (`gpt-5.6-luna`, `gpt-5.6-terra`) via the Responses API and **Gemini** (`gemini-3.8-flash`) via the Interactions API, both with strict JSON-schema structured output. Gemini can be used **free of charge** but is slower!
- ✅ Every subtitle block is validated; missing or malformed items are retried individually and an incomplete translation is never saved
- 🤪 Mock backend for **offline testing** (no token usage)
- 📂 Context menu support on video file:
  - Translate subtitles from `.srt` files or folders containing video files or **extracted from MKV** (currently experimental)
- 🔧 Configurable:
  - Target language (predefined or custom)
  - LLM provider (OpenAI or Gemini)
  - Model and API key selection
  - Token price estimation
  - Parallel request control (advanced setting)
- 📊 Live cost estimation before translation, using each model's input/output token prices (Gemini shows paid-tier prices; free-tier keys are not billed)
- 📊 Progress bar with cancel option and retries for rate limits and transient errors (honoring `Retry-After`)

---

## 🛠️ Requirements

- ✅ Kodi 20+
- ✅ API access to one or more of:
  - **OpenAI** → [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
  - **Gemini** → [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- ⚠️ You are responsible for your **own API usage and associated costs**.

---

## 📸 Screenshots

### Configuration – Language Selection
![Language Configuration](resources/screenshots/configuration_langugage.png)

### Configuration – Model, Provider and API Key
![Model Configuration](resources/screenshots/configuration_model.png)

### Translate from File Selector
When you run the add-on file selector lets you choose `.srt` file.
![File Selector](resources/screenshots/translate_file_selector.png)

### Translate from Context Menu
![Context Menu](resources/screenshots/translate_context_menu.png)

### Estimated Cost Dialog
![Cost Estimation](resources/screenshots/cost_estimation.png)

---

## 🚀 Installation and usage

1. [Download the latest `.zip` release](https://github.com/re999/script.program.sub-ai-translator/releases)
2. In Kodi:
   - Go to **Add-ons → Install from zip file**
   - Select the downloaded `.zip`
3. Configure the add-on by providing your API key(s) and language settings. **If using OpenAI key you also need to pre-paid some money** to make it work. Gemini can be used free of charge.
4. Open the add-on via:
   - **Program Add-ons → Subtitle AI Translator**
   - Or by right-clicking a video or subtitle file → **Translate subtitles**

---

## ⚙️ Configuration options

Accessible via **Add-on Settings**:

| Setting | Description |
|--------|-------------|
| **Target Language** | Choose a predefined language or enter a custom one |
| **Provider** | Select between OpenAI, Gemini or mock backend |
| **Model** | Choose supported model for selected provider. Legacy selections (`gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, Gemini 1.5/2.0, Auto) keep working and are mapped to current models |
| **API Key** | Paste your API key (OpenAI or Gemini) here |
| **OpenAI price override** | Optional blended USD price per 1000 tokens for the cost estimate. `0` (or the old default `0.001`) uses the built-in model prices |
| **Parallel Requests** | Number of batches translated concurrently (1–10). Gemini is capped at 2 to limit rate-limit (429) errors on free-tier keys |
| **Mock Backend** | Use fake responses for testing (no real API calls) |

---

## 🧪 Development

Run the automated tests (no API keys or network needed):

```bash
python3 -m pytest
```

Optional live smoke test against the real APIs (uses your account and costs tokens):

```bash
OPENAI_API_KEY=... GEMINI_API_KEY=... python3 tests/live_smoke.py [openai] [gemini]
```

---

## 💡 Roadmap

- [ ] Pause/resume translations
- [ ] Additional LLM backends (Mistral, Claude, local models, etc.)
- [ ] GUI improvements and setup wizard
- [ ] Format support beyond `.srt` (e.g., `.ass`, embedded subtitles)

---

## 🤝 Support This Project

If you find this add-on useful:

- 🌟 Star it on GitHub
- 🤍 Spread the word and give feedback
- 🐛 Report bugs or suggest features

---

## 📜 License

This project is licensed under the **MIT License**, see the `LICENSE` file for details, with the following addition:

> 📊 **Disclaimer**: You are fully responsible for any costs incurred by using this add-on. The author is not liable for API charges or misuse. Use at your own risk.

---

**© 2025 by re999**


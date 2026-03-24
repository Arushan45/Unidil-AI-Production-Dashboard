# Unidil AI Production Dashboard

Streamlit dashboard for UNIDIL production monitoring with:
- Daily production charts (`Corrugator`, `Tuber`)
- CSV + Google Sheet data merge
- Gemini-powered AI assistant

## 1. Run Locally

From project root:

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

Open: `http://localhost:8501`

## 2. Configure Gemini API Key (Local)

Set environment variable:

```powershell
$env:GOOGLE_API_KEY="your_gemini_api_key"
python -m streamlit run app.py
```

The app reads key from:
1. `st.secrets["GOOGLE_API_KEY"]` (Streamlit Cloud)
2. `GOOGLE_API_KEY` env var (local fallback)

## 3. Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Cloud -> `New app`.
3. Select repo and set main file path to `app.py`.
4. In app `Settings` -> `Secrets`, add:

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
```

5. Save and redeploy.

## 4. Security Notes

- Do not hardcode API keys in code.
- Keep `.streamlit/secrets.toml` out of Git (already in `.gitignore`).
- If a key was exposed, rotate/regenerate it and use a new one.

## 5. Data Source

- Local base file: `Daily Production Data_NEW Format - UNIDIL.csv`
- Live source: Google Sheet export (`UNIDIL` block only)


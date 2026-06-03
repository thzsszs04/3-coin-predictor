# 3-Coin Predictor

Streamlit dashboard for a graduation thesis project comparing SARIMAX and XGBoost models for cryptocurrency price prediction.

## Features

- Live market overview for Bitcoin, Ethereum, and Solana
- Prediction page with actual vs predicted charts, metrics, forecast horizon, model diagnostics, and AI Insight
- Backtesting simulation page
- Model performance comparison page
- Dataset, variable, and method explanation pages
- Indonesian and English language switch
- Modern quant-style responsive UI

## Run Locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Secrets

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.0-flash"
FAST_SARIMAX = "1"
```

Do not commit `.streamlit/secrets.toml`. It is ignored by `.gitignore`.

## Precomputed Forecast Snapshot

The deployed app uses `data/precomputed_results.pkl` to load homepage prediction and model-comparison results quickly without retraining on every visitor click. The live market overview remains live, while backtesting remains dynamic because users can choose many historical dates.

To refresh the snapshot with current rolling market data:

```bash
python3 scripts/precompute_results.py
```

Commit the updated `data/precomputed_results.pkl` after regenerating it.

The snapshot is also refreshed automatically by GitHub Actions every day at 01:30 WIB. You can trigger the same refresh manually from the repository's Actions tab by running the `Refresh precomputed forecasts` workflow.

## Recommended Deployment

This project is a Streamlit app, so the recommended public deployment target is Streamlit Community Cloud.

1. Push this folder to a GitHub repository.
2. Go to `https://share.streamlit.io`.
3. Create a new app from the GitHub repository.
4. Set the entrypoint file to:

```text
app.py
```

5. In Advanced settings, paste the Gemini secrets:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.0-flash"
FAST_SARIMAX = "1"
```

6. Deploy.

## Why Not Vercel?

Vercel is optimized for static sites and serverless functions. Streamlit needs a long-running interactive Python server with browser-session state and live UI communication, so Vercel is not a reliable target for this app.

Use Streamlit Community Cloud, Render, Railway, or a VM/container host instead.

# Telecom NOC Dashboard

Single-page Streamlit dashboard for telecom network performance monitoring.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Always launch this app using `run.ps1`, not `streamlit run app.py` directly. Launching directly bypasses the `__pycache__` cleanup step and can cause stale-cache ImportErrors to reappear.

The header logo requires `assets/tunisie_telecom_logo.webp` to be present in the repository.

## Structure

- `app.py` - dashboard entry point
- `backend/cleaning.py` - sample data generation and cleaning
- `backend/charts.py` - Plotly chart builders
- `backend/map.py` - map visualization
- `assets/` - icon assets

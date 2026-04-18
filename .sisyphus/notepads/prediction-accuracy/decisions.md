# Decisions — prediction-accuracy

## Architecture Decisions
- Fix scope: Comprehensive — RF inference fix + ConvLSTM implementation + all data issues
- Channel count: 8 channels (matching current data pipeline, fixing Train_ConvLSTM.py's outdated CHANNELS=6)
- API target: Only api_server.py (Flask) — the production path. FastAPI server left as-is.
- Risk/probability: RF probability used for probability field; temperature thresholds kept for risk levels
- Normalization: Temporal-aware split (compute stats on train only), requires retraining
- ConvLSTM hidden dims: [32, 32] (matching existing Train_ConvLSTM.py contract)
- Physics loss lambda: 0.1 (matching existing contract)

## Session History
- Session ses_338f35470ffeEmD3V5RJ5jPI0H: Started work session, updated boulder.json to prediction-accuracy plan

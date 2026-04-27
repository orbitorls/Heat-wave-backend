# Issues & Gotchas — prediction-accuracy

## Known Issues (Pre-Fix)
1. RF inference fakes temperature: `last_temp + (prob × 1.5°C)` in api_server.py lines 837-855
2. Hardcoded dummy values: `"date": "2000-XX-XX"` (line 875), `RH2M: 60, WS10M: 2.5` (lines 885-886), `PRECTOTCORR: 0, WS10M: 2, RH2M: 60, NDVI: 0.5` (lines 937-940)
3. Risk/probability ignores model output (lines 240-255): static lookup table
4. Normalization data leakage: stats on ALL data before split
5. Split mismatch: training 70/15/15 vs inference 80/20
6. No clipping during inference (training uses percentile clipping)
7. ConvLSTM is completely unimplemented (3 None stubs)
8. Train_ConvLSTM.py uses CHANNELS=6 but data pipeline produces 8 channels

## File Encoding Note
- requirements.txt is UTF-16 encoded — may cause issues on some systems

## [2026-03-07] Task 11: ConvLSTM Checkpoint Load Failure
- `heatwave_convlstm_v1.pth` state_dict uses `cell_list.*` / `final_conv.*`
- `src/models/convlstm.py` HeatwaveConvLSTM uses `encoder_cells.*` / `output_conv.*`
- `api_server.py:678` calls `model.load_state_dict()` without `strict=False` or key remapping
- No try/except around this call -- server crashes instead of falling back to RF
- This means ConvLSTM path is broken for the existing v1 checkpoint
- RF v25 works fine as fallback once ConvLSTM checkpoint is removed from discovery

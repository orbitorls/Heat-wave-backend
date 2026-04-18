# SKILL: Balanced Random Forest for Heatwave Prediction

**Description:** Specialized workflow for training heatwave prediction models in tropical regions (Thailand) using ERA5 data and class-balancing techniques.

## 📝 Procedural Guidance

### Research Verification
- Always check `research_knowledge/AI_READ_ME.md` for threshold values (e.g., 95th percentile).
- Verify the use of `imblearn.ensemble.BalancedRandomForestClassifier`.

### Data Pre-processing
- Ensure `t2m` is converted to Celsius if stored in Kelvin (Subtract 273.15).
- Apply a rolling window of 3-5 days to define the "Heatwave" event class.
- Balance the classes using the model's internal bootstrapping mechanism.

### Model Evaluation Protocol
1. Calculate **Precision-Recall Curve**.
2. Report **F1-Score** per location.
3. Visualize **SHAP values** to explain feature importance (e.g., "Why did the model predict a heatwave in Saraburi but not in Bangkok?").

## 🛠️ Resources
- **Reference Code:** `data_loader.py` (Coordinate encoding).
- **Target Script:** `Train_BalancedForest.py` (To be implemented).
- **Metric Library:** `scikit-learn.metrics`, `imbalanced-learn`.

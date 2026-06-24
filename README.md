![ALT](images/pexels-photo-1481105.jpg)

### House Prediction

# 🏠 House Price Prediction

**[🔗 Try the live app →](REPLACE_WITH_YOUR_STREAMLIT_URL)**

An end-to-end regression project that predicts residential **price per unit area** from three
property features, compares three regression models, and is deployed as a live, interactive web app.

> **Data honesty note:** Built on the public **UCI _Real Estate Valuation_** dataset (Sindian
> District, New Taipei City). The *"distance to nearest train station"* feature is Taipei transit
> data, generalised here for a hypothetical scenario. This is a demonstration of an end-to-end ML
> workflow — **not** a production UK valuation tool.

---

## What it does

Enter three values — house age, distance to the nearest train station (metres), and the number of
nearby convenience stores — and the app returns a predicted price per unit area. The trained model
is served behind a Streamlit interface, so anyone can use it from a browser with no setup.

## Approach

- **One clean train/test split**, reused across every model, so the comparison is like-for-like.
- **Scaled pipelines** — each model wrapped in a `StandardScaler` + estimator `Pipeline`, so the
  scaler learns only from the training data (no leakage) and regularisation is meaningful.
- **Three models compared** on the same held-out test set: Linear, Ridge, and RidgeCV regression.

## Results

| Model    | MAE   | RMSE  | R²    |
|----------|-------|-------|-------|
| Linear   | 7.134 | 9.983 | 0.453 |
| Ridge    | 7.135 | 9.983 | 0.454 |
| RidgeCV  | 7.143 | 9.979 | 0.454 |

All three models landed within a fraction of a point of each other. Linear Regression had the lowest
MAE, and because the regularised models offered no meaningful improvement, the **simplest model was
selected on the principle of parsimony** — when models are statistically tied, the simpler one is
preferred: easier to interpret, faster, and less prone to overfitting.

## What the model learned

Because the features were standardised, the coefficients are directly comparable in size:

| Feature                  | Coefficient | Reading |
|--------------------------|-------------|---------|
| Distance to train station| **−6.78**   | Strongest driver — further away, lower price |
| Convenience stores       | **+3.82**   | More stores nearby, higher price |
| House age                | **−2.88**   | Older houses, slightly lower price |

## Caveats

The modest R² (~0.45) is honest and expected — three features can't fully explain house prices. A
real-world version would retrain on **local** data with richer features (floor area, bedrooms,
condition, property type) and monitor for data drift over time.

## Tech stack

Python · pandas · scikit-learn (Pipeline, StandardScaler, Linear/Ridge/RidgeCV) · joblib · Streamlit

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo structure

```
├── app.py                  # Streamlit web app
├── house_model.joblib      # trained pipeline (what the app loads)
├── requirements.txt        # dependencies
├── Real estate.csv         # source dataset
├── House_Prediction_*.ipynb  # full analysis notebook
└── README.md
```

---

*Built by Kojusoluwa (Teni) Olutade — [GitHub: ProTeni](https://github.com/ProTeni)*

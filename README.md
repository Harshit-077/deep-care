# DeepCare: Deep Learning Based Early Disease Detection and Risk Prediction

Course project (Group 14, SIH26139). DeepCare estimates the risk of diabetes or prediabetes from 21 self-reported health indicators. It compares a tuned deep learning model (Keras MLP) with classical baselines using metrics suited to imbalanced medical data, and ships a Streamlit demo that explains each estimate.

> **Disclaimer:** This is an educational project and not a medical device. It must not be used for diagnosis or treatment decisions.

## Headline results

Held-out test set (34,422 respondents, 15.3% positive, no-skill PR-AUC 0.153):

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8085 | 0.4147 | 0.639 | 0.360 | 0.461 |
| Random Forest | 0.8150 | 0.4348 | 0.637 | 0.375 | 0.472 |
| XGBoost | 0.8177 | 0.4409 | 0.618 | 0.383 | 0.473 |
| **Keras MLP** | 0.8177 | 0.4377 | 0.599 | 0.389 | 0.472 |

Recall, precision and F1 use the F1-optimal threshold chosen on the validation set. 95% bootstrap intervals are in the report (roughly ±0.006 for ROC-AUC and ±0.014 for PR-AUC).

- **The MLP ties XGBoost.** A paired bootstrap cannot separate them (PR-AUC difference -0.003, 95% CI -0.007 to +0.0003). Both are slightly but clearly better than logistic regression. Hyperparameter tuning barely moved any model, so the limit is the information in the 21 features, not model capacity.
- **Screening trade-off.** Thresholds chosen on validation to reach 80% recall transferred to the test set (recall 0.80). They flag about 39% of people, roughly 3.2 flagged people per true case found.

| Operating point (80% recall) | Precision | Share flagged | Flagged per case found |
|---|---|---|---|
| Logistic Regression | 0.305 | 39.9% | 3.28 |
| Random Forest | 0.309 | 39.7% | 3.24 |
| XGBoost | 0.315 | 38.7% | 3.17 |
| Keras MLP | 0.316 | 38.7% | 3.17 |

- **Calibration.** Class weighting inflated raw scores about 2.6 times (mean 0.40 vs 0.153 observed). Platt scaling fixed it (expected calibration error 0.25 to 0.003 for the MLP).
- **Explainability.** SHAP ranks general health, BMI, blood pressure, cholesterol and age highest, in agreement with XGBoost's TreeSHAP (Spearman 0.945).
- **Uneven coverage.** At one global cutoff, recall is 0.31 for ages 18-34 and 0.55 for ages 35-49, versus 0.80-0.86 from age 50, and 0.89 for the lowest income band versus 0.70 for the highest. Calibration holds within every group; the cutoff itself behaves like an age and income filter.

## Project status

- [x] Repository setup
- [x] Exploratory data analysis (`notebooks/01_eda.ipynb`)
- [x] Preprocessing and train/validation/test split (`notebooks/02_preprocessing.ipynb`)
- [x] Baseline models (`notebooks/03_baselines.ipynb`)
- [x] Keras MLP with Keras Tuner and TensorBoard (`notebooks/04_mlp.ipynb`)
- [x] Evaluation: bootstrap intervals, operating points, calibration (`notebooks/05_evaluation.ipynb`)
- [x] Explainability and subgroup analysis (`notebooks/06_explainability.ipynb`)
- [x] Streamlit demo app (`notebooks/07_export_for_app.ipynb`, `app/`)
- [ ] Final report: export the PDF into `reports/`

## Dataset

**CDC Diabetes Health Indicators**, UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Citation: *CDC Diabetes Health Indicators* [Dataset]. (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919
- 253,680 survey responses, 21 features, no missing values
- Target `Diabetes_binary`: 0 = no diabetes, 1 = prediabetes or diabetes
- Features: health conditions (blood pressure, cholesterol, stroke, heart disease, walking difficulty), self-rated health (general, mental and physical), body and lifestyle (BMI, smoking, activity, diet, alcohol), healthcare access, and demographics (sex, age group, education, income)

| Finding | Value |
|---|---|
| Raw rows | 253,680 |
| Exact duplicate rows | 24,206 (9.5%) |
| Rows after deduplication | 229,474 |
| Positive rate (raw / deduplicated) | 13.9% / 15.3% |
| Feature profiles with conflicting labels | 1,566 |
| Split (train / validation / test) | 160,631 / 34,421 / 34,422 |

The data is not committed to this repository. Notebook 01 downloads it with the `ucimlrepo` package and caches it in Google Drive.

## Methodology

1. **Preprocessing:** drop exact duplicates before splitting (prevents the same record landing in train and test); stratified 70/15/15 split (seed 42); clip BMI at training-set percentiles (17.0 to 56.0); standardize the seven non-binary features with a scaler fit on the training set only; leave the 14 binary features as 0/1.
2. **Imbalance:** class-weighted losses rather than synthetic oversampling, because the features are binary and ordinal.
3. **Models:** Logistic Regression, Random Forest, XGBoost, and a Keras MLP. The tuned MLP has three hidden layers (64, 128, 128) with batch normalization, ReLU and dropout 0.2.
4. **Protocol (identical for all models):** hyperparameters chosen by random search on validation PR-AUC; decision threshold chosen on the validation set; test set used once; validation and test probabilities saved so every model is compared on identical data.
5. **Evaluation:** ROC-AUC, PR-AUC, recall, precision, specificity, F1, paired bootstrap confidence intervals, thresholds for target recall, calibration (Brier score, expected calibration error), SHAP, and subgroup analysis by sex, age band and income band.

Accuracy is deliberately not the headline metric: predicting "no diabetes" for everyone already scores about 85%.

## Demo app

A Streamlit app (`app/app.py`) takes the 21 survey answers and shows the calibrated risk, whether it is above or below the 80%-recall cutoff (about 15% risk), and a SHAP chart of what moved the estimate. It runs the trained MLP as plain NumPy (no TensorFlow needed); notebook 07 exports the weights and checks that the NumPy pipeline reproduces the Keras model.

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app reads four files from `models/`: `mlp_weights.npz`, `app_bundle.json`, `app_config.json`, `shap_background.npz` (all produced by notebooks 06 and 07). It can be deployed on Streamlit Community Cloud with `app/app.py` as the main file.

## Repository structure

```
deepcare/
├── README.md
├── requirements.txt       # notebook environment
├── notebooks/             # 01_eda ... 07_export_for_app, run in numbered order
├── app/                   # app.py, inference.py, requirements.txt
├── models/                # files the demo app needs (small)
├── reports/               # final report, key figures and result tables
└── data/                  # not committed; downloaded by the notebooks
```

## Reproducing the experiments

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshit-077/deepcare.git
   cd deepcare
   ```
2. Open the notebooks in [Google Colab](https://colab.research.google.com/) (File, Open notebook, GitHub), or run locally after `pip install -r requirements.txt`.
3. Run the notebooks in order. They mount Google Drive and keep data, trained models, results and figures under `/content/drive/MyDrive/deepcare/`, so outputs survive Colab session resets.
4. Everything uses seed 42. Experiments were run on Google Colab with TensorFlow 2.21.0; exact package versions are pinned in `requirements.txt`.

## Tech stack

Python, TensorFlow/Keras, Keras Tuner, TensorBoard, scikit-learn, XGBoost, SHAP, pandas, NumPy, matplotlib/seaborn, Streamlit, Google Colab.

## Limitations

- Self-reported survey data, not clinical measurements: this supports population-level risk screening, not clinical early detection. The positive class combines prediabetes and diabetes.
- Deduplication changes the class balance (13.9% to 15.3%), so precision and PR-AUC are not directly comparable to results on the raw data.
- Calibrated probabilities describe risk within this dataset, not population risk. The dataset carries no survey weights.
- 1,566 feature profiles appear with conflicting labels, which caps achievable performance.
- One data split and no external validation; bootstrap intervals capture test-sampling uncertainty only.
- A single cutoff gives uneven recall across age and income groups (see above).
- SHAP shows what the model relies on, not what causes diabetes. Some associations (for example heavy alcohol use and skipped cholesterol checks lowering predicted risk) reflect survey and screening effects, not protective factors.

## Author

Harshit

## Acknowledgements

Dataset provided by the CDC Behavioral Risk Factor Surveillance System via the UCI Machine Learning Repository.
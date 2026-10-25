# DeepCare: Deep Learning Based Early Disease Detection and Risk Prediction

Course project (Group 14, SIH26139). DeepCare predicts an individual's risk of diabetes from health and lifestyle survey indicators, comparing a deep learning model (Keras MLP) against classical machine learning baselines using classification metrics suited to imbalanced medical data.

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Project status

- [x] Repository setup
- [x] Exploratory data analysis (`notebooks/01_eda.ipynb`)
- [ ] Preprocessing and train/validation/test split (`notebooks/02_preprocessing.ipynb`)
- [ ] Baseline models (`notebooks/03_baselines.ipynb`)
- [ ] Keras MLP with hyperparameter tuning (`notebooks/04_mlp.ipynb`)
- [ ] Evaluation, calibration and SHAP explainability
- [ ] Streamlit demo app
- [ ] Final report

## Problem statement

Use a labelled health dataset with relevant patient features, preprocess the data, build a deep learning disease-risk prediction model, compare its performance against other approaches, and evaluate it with suitable classification metrics.

## Dataset

**CDC Diabetes Health Indicators**, from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Citation: *CDC Diabetes Health Indicators* [Dataset]. (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919
- 253,680 survey responses, 21 features, no missing values
- Target `Diabetes_binary`: 0 = no diabetes, 1 = prediabetes or diabetes
- Features cover demographics (age group, sex, education, income), lifestyle (smoking, physical activity, diet, alcohol), health history (blood pressure, cholesterol, stroke, heart disease) and self-reported general, mental and physical health

Findings from our exploratory analysis:

| Item | Value |
|---|---|
| Raw rows | 253,680 |
| Exact duplicate rows | 24,206 (9.5%) |
| Rows after deduplication | 229,474 |
| Positive rate (raw / deduplicated) | 13.9% / 15.3% |
| Feature profiles with conflicting labels | 1,566 |

The data is not committed to this repository. Notebooks download it with the `ucimlrepo` package and cache it in Google Drive.

## Methodology

1. **Exploration:** class balance, duplicates, feature distributions, relationship of each feature with the target.
2. **Preprocessing:**
   - Drop exact duplicate rows before splitting, to prevent identical records appearing in both train and test sets.
   - Stratified 70/15/15 train/validation/test split with a fixed seed.
   - Clip BMI at training-set percentiles, standardize non-binary features, and fit all preprocessing on the training set only.
3. **Imbalance handling:** class-weighted loss rather than synthetic oversampling, because most features are binary or ordinal.
4. **Models compared:**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Keras MLP (batch normalization, dropout, class weights, early stopping), tuned with Keras Tuner and tracked with TensorBoard
5. **Evaluation:** recall (sensitivity), precision, specificity, F1, ROC-AUC, PR-AUC, confusion matrix, calibration curve, and a decision threshold selected on the validation set. The test set is used once, for the final comparison.
6. **Explainability:** SHAP feature attributions and a subgroup analysis by age band and sex.

Accuracy is deliberately not the headline metric: predicting "no diabetes" for everyone already scores about 85%.

## Results

_To be completed after experiments._

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| Keras MLP | | | | | |

## Repository structure

```
deepcare/
├── README.md
├── requirements.txt
├── notebooks/     # Colab notebooks, run in numbered order
├── models/        # saved model and preprocessing artifacts (small files)
├── app/           # Streamlit demo
├── reports/       # figures, result tables, final report
└── data/          # not committed; downloaded by the notebooks
```

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshit-077/deepcare.git
   cd deepcare
   ```
2. Open the notebooks in [Google Colab](https://colab.research.google.com/) (File → Open notebook → GitHub), or run locally after installing dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order (`01_eda`, `02_preprocessing`, ...). Each notebook mounts Google Drive and saves its outputs there so results survive Colab session resets.
4. Run the demo app (once added):
   ```bash
   streamlit run app/app.py
   ```

## Tech stack

Python, TensorFlow/Keras, scikit-learn, XGBoost, Keras Tuner, TensorBoard, SHAP, pandas, matplotlib/seaborn, Streamlit, Google Colab.

## Limitations

- The data is self-reported survey data, not clinical measurements, so the model supports population-level risk screening rather than clinical early detection.
- The positive class combines prediabetes and diabetes.
- Deduplication changes the class balance (13.9% to 15.3%), so precision and PR-AUC are not directly comparable to results on the raw data.
- 1,566 feature profiles appear with conflicting labels, which places a ceiling on achievable performance.
- No external validation set; results reflect a single survey dataset and may not generalize to other populations.
- Features such as sex, age, income and education can encode demographic bias; subgroup performance is reported separately.

## Author

Harshit

## Acknowledgements

Dataset provided by the CDC Behavioral Risk Factor Surveillance System via the UCI Machine Learning Repository.
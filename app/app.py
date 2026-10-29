"""DeepCare demo: estimate diabetes/prediabetes risk from survey-style health indicators.

Educational demo for a course project. Not a medical device and not medical advice.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

import inference

st.set_page_config(page_title='DeepCare', page_icon='🩺', layout='wide')

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


@st.cache_resource
def load_artifacts():
    return inference.load_bundle(MODELS_DIR)


art = load_artifacts()
cfg = art['cfg']
ORDER = cfg['feature_order']
PREVALENCE = cfg['dataset_prevalence']
CUTOFF = cfg['threshold_recall80_calibrated']

AGE = {1: '18–24', 2: '25–29', 3: '30–34', 4: '35–39', 5: '40–44', 6: '45–49', 7: '50–54',
       8: '55–59', 9: '60–64', 10: '65–69', 11: '70–74', 12: '75–79', 13: '80 or older'}
EDU = {1: 'Never attended school or kindergarten only', 2: 'Grades 1–8', 3: 'Grades 9–11',
       4: 'Grade 12 or GED', 5: 'Some college or technical school', 6: 'College graduate'}
INCOME = {1: 'Less than $10,000', 2: '$10,000 to under $15,000', 3: '$15,000 to under $20,000',
          4: '$20,000 to under $25,000', 5: '$25,000 to under $35,000', 6: '$35,000 to under $50,000',
          7: '$50,000 to under $75,000', 8: '$75,000 or more'}
GENHLTH = {1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair', 5: 'Poor'}

NAMES = {'HighBP': 'High blood pressure', 'HighChol': 'High cholesterol', 'CholCheck': 'Cholesterol check (5 yrs)',
         'BMI': 'BMI', 'Smoker': 'Smoked 100+ cigarettes', 'Stroke': 'Ever had a stroke',
         'HeartDiseaseorAttack': 'Heart disease / heart attack', 'PhysActivity': 'Physical activity (30 days)',
         'Fruits': 'Fruit daily', 'Veggies': 'Vegetables daily', 'HvyAlcoholConsump': 'Heavy drinking',
         'AnyHealthcare': 'Has health coverage', 'NoDocbcCost': 'Skipped doctor due to cost',
         'GenHlth': 'General health', 'MentHlth': 'Poor mental health days', 'PhysHlth': 'Poor physical health days',
         'DiffWalk': 'Difficulty walking', 'Sex': 'Sex', 'Age': 'Age group', 'Education': 'Education',
         'Income': 'Income'}


def describe(feature, value):
    v = int(round(value)) if feature != 'BMI' else value
    if feature == 'BMI':
        return f'BMI = {value:.1f}'
    if feature == 'GenHlth':
        return f'General health = {GENHLTH[v]}'
    if feature == 'Age':
        return f'Age group = {AGE[v]}'
    if feature == 'Income':
        return f'Income = {INCOME[v]}'
    if feature == 'Education':
        return f'Education = {EDU[v]}'
    if feature == 'Sex':
        return f'Sex = {"Male" if v == 1 else "Female"}'
    if feature in ('MentHlth', 'PhysHlth'):
        return f'{NAMES[feature]} = {v}'
    return f'{NAMES[feature]} = {"Yes" if v == 1 else "No"}'


# ----------------------------------------------------------------------------- inputs
st.title('DeepCare: diabetes risk estimator')
st.caption('Educational demo for a deep learning course project (Keras MLP trained on the CDC Diabetes Health '
           'Indicators survey data). Not a medical device and not medical advice.')

vals = {}
with st.sidebar.form('person'):
    st.header('Your answers')
    st.subheader('About you')
    vals['Sex'] = 1 if st.selectbox('Sex', ['Female', 'Male']) == 'Male' else 0
    vals['Age'] = st.selectbox('Age group', list(AGE), index=7, format_func=AGE.get)
    vals['Education'] = st.selectbox('Education', list(EDU), index=4, format_func=EDU.get)
    vals['Income'] = st.selectbox('Household income', list(INCOME), index=5, format_func=INCOME.get)
    lo, hi = cfg['feature_ranges']['BMI']
    vals['BMI'] = st.number_input('BMI', min_value=float(lo), max_value=float(hi), value=27.0, step=0.5)

    st.subheader('Health')
    vals['GenHlth'] = st.select_slider('General health', options=list(GENHLTH), value=3, format_func=GENHLTH.get)
    vals['MentHlth'] = st.slider('Days of poor mental health (last 30)', 0, 30, 0)
    vals['PhysHlth'] = st.slider('Days of poor physical health (last 30)', 0, 30, 0)
    vals['HighBP'] = int(st.checkbox('Told you have high blood pressure'))
    vals['HighChol'] = int(st.checkbox('Told you have high cholesterol'))
    vals['CholCheck'] = int(st.checkbox('Cholesterol checked in the past 5 years', value=True))
    vals['Stroke'] = int(st.checkbox('Ever told you had a stroke'))
    vals['HeartDiseaseorAttack'] = int(st.checkbox('Coronary heart disease or heart attack'))
    vals['DiffWalk'] = int(st.checkbox('Serious difficulty walking or climbing stairs'))

    st.subheader('Lifestyle and access')
    vals['Smoker'] = int(st.checkbox('Smoked at least 100 cigarettes in your life'))
    vals['PhysActivity'] = int(st.checkbox('Physical activity in the past 30 days (outside work)', value=True))
    vals['Fruits'] = int(st.checkbox('Eat fruit at least once a day', value=True))
    vals['Veggies'] = int(st.checkbox('Eat vegetables at least once a day', value=True))
    vals['HvyAlcoholConsump'] = int(st.checkbox('Heavy drinker (men over 14, women over 7 drinks a week)'))
    vals['AnyHealthcare'] = int(st.checkbox('Have health care coverage', value=True))
    vals['NoDocbcCost'] = int(st.checkbox('Needed a doctor in the past year but could not afford it'))
    st.form_submit_button('Estimate risk', type='primary')

assert set(vals) == set(ORDER), 'form fields do not match the model features'

# ----------------------------------------------------------------------------- prediction
x_raw = np.array([[vals[f] for f in ORDER]], dtype='float64')
x = inference.preprocess(x_raw, art)
risk = float(inference.predict_risk(x, art)[0])


@st.cache_data(show_spinner='Computing explanation...')
def explain(x_key):
    row = np.array([x_key], dtype='float64')

    def f(z):
        return inference.predict_risk(np.asarray(z, dtype='float64'), art)

    np.random.seed(0)
    explainer = shap.KernelExplainer(f, art['background'])
    sv = np.array(explainer.shap_values(row, nsamples=300, silent=True))
    if sv.ndim == 3:
        sv = sv[..., 0]
    return sv.reshape(-1), float(np.ravel(explainer.expected_value)[0])


sv, reference = explain(tuple(np.round(x[0], 6)))

# ----------------------------------------------------------------------------- results
left, right = st.columns([1, 1.4])
with left:
    st.subheader('Estimated risk')
    st.metric('Probability of diabetes or prediabetes', f'{risk:.1%}',
              delta=f'{risk / PREVALENCE:.1f}x the survey average ({PREVALENCE:.1%})', delta_color='off')
    if risk >= CUTOFF:
        st.warning(f'**Above the screening cutoff ({CUTOFF:.0%}).** People with answers like these are '
                   'flagged by the model. This is not a diagnosis; a blood test is the way to find out.')
    else:
        st.info(f'**Below the screening cutoff ({CUTOFF:.0%}).** This does not mean no risk: the cutoff '
                'misses about 1 in 5 cases overall, and more among younger adults.')
    if not (cfg['bmi_clip'][0] <= vals['BMI'] <= cfg['bmi_clip'][1]):
        st.caption(f"BMI outside {cfg['bmi_clip'][0]:.0f}–{cfg['bmi_clip'][1]:.0f} is treated as the nearest "
                   'bound by the model.')

with right:
    st.subheader('What moved this estimate')
    top = np.argsort(-np.abs(sv))[:8]
    labels = [describe(ORDER[i], vals[ORDER[i]]) for i in top][::-1]
    effects = (sv[top] * 100)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, effects, color=['#d62728' if e > 0 else '#1f77b4' for e in effects])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Effect on estimated risk (percentage points)')
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f'Starting point: {reference:.1%}, the average estimate for a reference group of survey '
               'respondents. Red bars raise the estimate, blue bars lower it. Explanations are SHAP estimates '
               'of what the model relies on, not causes.')

with st.expander('How to read this, and limitations'):
    st.markdown(f"""
- **What the number means.** It is the model's estimate of how often survey respondents with similar answers
  reported diabetes or prediabetes. The estimate is calibrated to this dataset, where {PREVALENCE:.1%} of
  respondents reported it. It is not a clinical probability.
- **Self-reported survey data.** The model learned from answers to a US health survey, not lab results.
  Some inputs act as proxies, for example not having had a cholesterol check lowers the estimate, which
  reflects who gets screened and diagnosed, not better health.
- **One cutoff, uneven coverage.** The {CUTOFF:.0%} cutoff catches about 80% of cases in the test data overall,
  but about 55% for ages 35 to 49 and about 31% for ages 18 to 34 (a small group), versus roughly 80% or
  more from age 50.
- **Moderate accuracy.** On held-out data the model separates cases from non-cases reasonably (ROC-AUC about
  0.82) but flagged people are mostly not cases: about 3 people are flagged for every case found.
- **Only 21 survey answers.** Family history, medications, lab values and many other factors are not included.
""")

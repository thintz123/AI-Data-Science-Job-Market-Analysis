import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("AI Job Market Dataset.csv")

if 'job_id' in df.columns:
    df = df.drop(columns=['job_id'])

cat_cols = ['job_title', 'company_size', 'company_industry', 'country', 'remote_type', 
            'experience_level', 'education_level', 'hiring_urgency']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop(columns=['salary'])
y = df['salary']
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42)
model.fit(X, y)

# Define common/mode values for a "Standard" Profile in 2026
# This creates a baseline for prediction
unique_titles_encoded = df['job_title'].unique()
standard_profiles_2026 = []

for title_enc in unique_titles_encoded:
    title_df = df[df['job_title'] == title_enc]
    row = {
        'job_title': title_enc,
        'company_size': df['company_size'].mode()[0],
        'company_industry': df['company_industry'].mode()[0],
        'country': df['country'].mode()[0],
        'remote_type': df['remote_type'].mode()[0],
        'experience_level': df['experience_level'].mode()[0],
        'years_experience': int(df['years_experience'].mean()),
        'education_level': df['education_level'].mode()[0],
        'skills_python': 1, # Assumed common skills
        'skills_sql': 1,
        'skills_ml': 1,
        'skills_deep_learning': 1,
        'skills_cloud': 1,
        'job_posting_month': 6,
        'job_posting_year': 2026,
        'hiring_urgency': df['hiring_urgency'].mode()[0],
        'job_openings': int(df['job_openings'].mean())
    }
    standard_profiles_2026.append(row)

profiles_2026_df = pd.DataFrame(standard_profiles_2026)[X.columns]
pred_2026 = model.predict(profiles_2026_df)

# Calculate Historical Trend for Each Title
# We use the raw dataset to find average growth per year per title
df_raw = pd.read_csv("AI Job Market Dataset.csv")
title_year_avg = df_raw.groupby(['job_title', 'job_posting_year'])['salary'].mean().unstack()
growth_rates = title_year_avg.pct_change(axis=1).mean(axis=1).fillna(0)

#Apply Growth Rate to 2026 Model Baseline to get 2027
forecast_2027 = []
for i, title_enc in enumerate(unique_titles_encoded):
    title_str = le_dict['job_title'].inverse_transform([title_enc])[0]
    growth = growth_rates.get(title_str, 0.0)
    salary_2026 = pred_2026[i]
    salary_2027 = salary_2026 * (1 + growth)
    
    forecast_2027.append({
        'Job Title': title_str,
        '2026 Baseline': round(salary_2026),
        'Growth Trend': f"{growth:+.2%}",
        'Predicted 2027 Salary': round(salary_2027)
    })

#Format Results
results_df = pd.DataFrame(forecast_2027).sort_values(by='Predicted 2027 Salary', ascending=False)
results_df.to_csv("avg_salary_prediction_2027.csv", index=False)
print(results_df)
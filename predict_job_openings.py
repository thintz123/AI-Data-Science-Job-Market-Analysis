import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('AI Job Market Dataset.csv')

# 1. Calculate historical sum of openings per job title per year
title_year_sum = df.groupby(['job_title', 'job_posting_year'])['job_openings'].sum().unstack()

# 2. Calculate YoY growth of the sum per title
sum_growth_rates = title_year_sum.pct_change(axis=1).mean(axis=1).fillna(0)

#Get the most recent sum as baseline (2026)
if 2026 in title_year_sum.columns:
    baseline_sum_2026 = title_year_sum[2026]
else:
    # Fallback to the latest available year if 2026 isn't the only one
    latest_year = title_year_sum.columns[-1]
    baseline_sum_2026 = title_year_sum[latest_year]

# Predict 2027 by applying the sum growth rate to the 2026 baseline
forecast_2027_sum = []
for title in title_year_sum.index:
    growth = sum_growth_rates.get(title, 0.0)
    total_2026 = baseline_sum_2026.get(title, 0.0)
    total_2027 = total_2026 * (1 + growth)
    
    forecast_2027_sum.append({
        'Job Title': title,
        '2026 Total Openings': round(total_2026),
        'Market Growth Trend': f"{growth:+.2%}",
        'Predicted 2027 Total Openings': round(total_2027)
    })

# Format results
demand_sum_results = pd.DataFrame(forecast_2027_sum).sort_values(by='Predicted 2027 Total Openings', ascending=False)

#Visualization for Demand Sum
plt.figure(figsize=(10, 6))
plt.barh(demand_sum_results['Job Title'], demand_sum_results['Predicted 2027 Total Openings'], color='navy')
plt.title('Predicted Total Market Openings by Job Title in 2027')
plt.xlabel('Estimated Total Openings across Industry')
plt.tight_layout()
plt.savefig('total_demand_forecast_2027.png')

print(demand_sum_results)
demand_sum_results.to_csv('total_job_demand_2027.csv', index=False)
# AI Data Science Job Market Analysis

A data science project that analyzes and forecasts trends in the AI and Data Science job market using machine learning models and Power BI to visualize the data.

## Project Overview

This project analyzes historical AI job market data to:
- **Predict job opening demand** by role for 2027
- **Forecast salary trends** across different job titles
- **Identify market growth patterns** to guide career decisions

## Key Features

- **Job Demand Forecasting**: Uses year-over-year growth rates to predict total market openings for each job title
- **Salary Prediction**: Employs Gradient Boosting Regression to model salary trends based on multiple factors
- **Data Visualization**: Generates charts showing predicted market demand
- **CSV Exports**: Produces detailed forecast reports for further analysis

## Dataset

The analysis uses `AI Job Market Dataset.csv` containing:
- Job titles, company information, and hiring details
- Salary data, experience requirements, and skills
- Job posting dates and remote work types
- Historical data spanning multiple years

## Files

- `predict_job_openings.py` - Forecasts total job openings by title for 2027
- `predict_salary.py` - Predicts average salary trends by job title
- `AI Job Market Dataset.csv` - Input dataset
- `total_job_demand_2027.csv` - Demand forecast results
- `avg_salary_prediction_2027.csv` - Salary forecast results
- `total_demand_forecast_2027.png` - Visualization of job demand

## Installation

```bash
# Clone the repository
git clone https://github.com/thintz123/AI-Data-Science-Job-Market-Analysis.git
cd AI-Data-Science-Job-Market-Analysis

# Install required dependencies
pip install pandas numpy scikit-learn matplotlib

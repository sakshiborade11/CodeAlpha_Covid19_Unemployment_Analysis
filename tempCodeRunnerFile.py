import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset with correct filename
df = pd.read_csv("Unemployment_in_India.csv")

# Clean column names immediately (spaces -> underscores, lowercase)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Initial inspection
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()

# Drop rows with any missing values
df = df.dropna()

# Set plot style
sns.set_style("whitegrid")

# Overall unemployment rate trend (all India)
plt.figure(figsize=(12, 6))
overall_trend = df.groupby('date')['estimated_unemployment_rate_(%)'].mean()
plt.plot(overall_trend.index, overall_trend.values)
plt.title('Average Unemployment Rate (%) in India Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Urban vs Rural trend
plt.figure(figsize=(12, 6))
urban = df[df['area'] == 'Urban'].groupby('date')['estimated_unemployment_rate_(%)'].mean()
rural = df[df['area'] == 'Rural'].groupby('date')['estimated_unemployment_rate_(%)'].mean()
plt.plot(urban.index, urban.values, label='Urban')
plt.plot(rural.index, rural.values, label='Rural')
plt.title('Unemployment Rate: Urban vs Rural')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Covid-19 impact analysis
before_covid = df[df['date'] < '2020-03-01']['estimated_unemployment_rate_(%)'].mean()
during_covid = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2020-06-30')]['estimated_unemployment_rate_(%)'].mean()
print(f"\nMean Unemployment Rate before Covid (before Mar 2020): {before_covid:.2f}%")
print(f"Mean Unemployment Rate during Covid (Mar 2020 - Jun 2020): {during_covid:.2f}%")

# Top 5 states by average unemployment rate
top_states = df.groupby('region')['estimated_unemployment_rate_(%)'].mean().sort_values(ascending=False).head(5)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_states.index, y=top_states.values, palette='viridis')
plt.title('Top 5 States by Average Unemployment Rate (%)')
plt.ylabel('Average Unemployment Rate (%)')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Seasonality: Monthly average unemployment rate for India
df['month'] = df['date'].dt.month
monthly_avg = df.groupby('month')['estimated_unemployment_rate_(%)'].mean()
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.title('Average Unemployment Rate (%) by Month (Seasonality Check)')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

# Heatmap: state vs. month (seasonality & regional variation)
df['year_month'] = df['date'].dt.to_period('M')
heatmap_data = df.pivot_table(
    index='region',
    columns='year_month',
    values='estimated_unemployment_rate_(%)',
    aggfunc='mean'
)
plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, cmap='RdYlBu_r', linewidths=0.5)
plt.title('Unemployment Rate (%) by State and Month')
plt.ylabel('State')
plt.xlabel('Year-Month')
plt.tight_layout()
plt.show()

# Insights: Min, mean, max unemployment per state
summary = df.groupby('region')['estimated_unemployment_rate_(%)'].agg(['min', 'mean', 'max']).sort_values('mean', ascending=False)
print("\nState-wise Unemployment Rate Summary (min, mean, max):")
print(summary)

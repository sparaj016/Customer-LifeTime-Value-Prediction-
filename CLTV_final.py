import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap

# Dataset
file_path = 'data.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Missing CustomerID rows and duplicates
df.dropna(subset=['CustomerID'], inplace=True)
df.drop_duplicates(inplace=True)

# Negative or zero quantity and prices (canceled or invalid orders)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Unused columns
df.drop(columns=['StockCode', 'Description', 'Country'], inplace=True, errors='ignore')

# Clean and format
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['customerid'] = df['customerid'].astype(str)
df.sort_values(by=['invoicedate', 'invoiceno'], inplace=True)

# Additional features for seasonal trend analysis
df['month'] = df['invoicedate'].dt.month.apply(lambda x: calendar.month_name[x])
df['dayofweek'] = df['invoicedate'].dt.day_name()

# Revenue summaries
month_order = list(calendar.month_name)[1:]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

monthly_revenue = df.groupby('month')['totalprice'].sum().reindex(month_order)
weekday_revenue = df.groupby('dayofweek')['totalprice'].sum().reindex(day_order)

print("\nMonthly Revenue:")
print(monthly_revenue)
print("\nRevenue by Day of Week:")
print(weekday_revenue)

# Snapshot date (1 day after last purchase)
snapshot_date = df['invoicedate'].max() + timedelta(days=1)

# Features per Customer
customer_df = df.groupby('customerid').agg({
    'invoicedate': [lambda x: (snapshot_date - x.max()).days,  # Recency
                    lambda x: (x.max() - x.min()).days],       # Tenure
    'invoiceno': 'nunique',                                    # Frequency
    'totalprice': 'sum'                                        # Monetary
}).reset_index()

# Rename columns
customer_df.columns = ['customerid', 'recency', 'tenure', 'frequency', 'monetary']

# AOV (Average Order Value)
customer_df['aov'] = customer_df['monetary'] / customer_df['frequency']

# Handle non-unique bin edges for RFM Segmentation
customer_df['r_score'] = pd.qcut(customer_df['recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
customer_df['f_score'] = pd.qcut(customer_df['frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
customer_df['m_score'] = pd.qcut(customer_df['monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
customer_df['rfm_score'] = customer_df['r_score'].astype(str) + customer_df['f_score'].astype(str) + customer_df['m_score'].astype(str)

print("\nTop RFM Scores:")
print(customer_df['rfm_score'].value_counts().head())

# Features and target
features = ['recency', 'tenure', 'frequency', 'aov']
target = 'monetary'
X = customer_df[features]
y = customer_df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
print("\nXGBoost Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)

# Predict full dataset and segment
customer_df['ltv_predicted'] = model.predict(X)
customer_df['segment'] = pd.qcut(customer_df['ltv_predicted'].rank(method="first"), 4, labels=['Low', 'Medium', 'High', 'Top'])

print("\nSample Predictions:")
print(customer_df[['customerid', 'ltv_predicted', 'segment', 'rfm_score']].head())

# Plot distribution of predicted LTV
sns.histplot(customer_df['ltv_predicted'], bins=30, kde=True)
plt.title("Predicted Customer Lifetime Value Distribution")
plt.xlabel("LTV")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.savefig("ltv_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(customer_df[['recency', 'frequency', 'tenure', 'aov', 'monetary']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("feature_correlation_matrix.png")
plt.show()

# SHAP
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.show()

# Results
customer_df.to_csv("cltv_predictions.csv", index=False)
print("\nExported 'cltv_predictions.csv' and saved visualizations.")

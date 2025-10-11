import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = 'ProjectML/Superstore.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')
df['Delivery Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

category_profit = df.groupby(['Category', 'Sub-Category'])[['Sales', 'Profit']].sum().reset_index()
print("1. Profit margins by category and sub-category")
print(category_profit)
plt.figure(figsize=(12, 6))
sns.barplot(data=category_profit, x='Sub-Category', y='Profit', hue='Category')
plt.title('Profit by Category and Sub-Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

state_profit = df.groupby('State')[['Sales', 'Profit']].sum().reset_index()
print("\n2. Sales and profit by state")
print(state_profit.sort_values(by='Sales', ascending=False).head())
plt.figure(figsize=(12, 6))
top_states = state_profit.sort_values(by='Sales', ascending=False).head(10)
sns.barplot(data=top_states, x='State', y='Sales')
plt.title('Top 10 States by Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

discount_impact = df.groupby('Discount')[['Sales', 'Profit']].mean().reset_index()
print("\n3. Average Sales and Profit by Discount")
print(discount_impact)
plt.figure(figsize=(10, 5))
sns.lineplot(data=discount_impact, x='Discount', y='Profit', label='Profit')
sns.lineplot(data=discount_impact, x='Discount', y='Sales', label='Sales')
plt.title('Impact of Discount on Sales and Profit')
plt.tight_layout()
plt.show()

product_loss = df.groupby('Product Name')[['Sales', 'Profit']].sum().sort_values(by='Profit').head(10).reset_index()
print("\n4. Top 10 Loss-Making Products")
print(product_loss)

delivery_by_mode = df.groupby('Ship Mode')['Delivery Time'].mean().reset_index()
print("\n5. Average Delivery Time by Shipping Mode")
print(delivery_by_mode)
plt.figure(figsize=(8, 5))
sns.barplot(data=delivery_by_mode, x='Ship Mode', y='Delivery Time')
plt.title('Average Delivery Time by Shipping Mode')
plt.tight_layout()
plt.show()

shipmode_profit = df.groupby('Ship Mode')[['Sales', 'Profit']].sum().reset_index()
print("\n6. Profit by Shipping Mode")
print(shipmode_profit)

segment_order = df.groupby('Segment').agg({'Sales': 'mean', 'Order ID': 'count'}).reset_index()
print("\n8. Average Order Value by Segment")
print(segment_order)

top_customers = df.groupby(['Customer ID', 'Customer Name'])[['Profit']].sum().sort_values(by='Profit', ascending=False).head(10).reset_index()
print("\n9. Top 10 Profitable Customers")
print(top_customers)

repeat_customers = df.groupby('Customer ID').agg({
    'Order ID': 'nunique',
    'Discount': 'mean',
    'Profit': 'sum'
}).reset_index()
print("\n10. Repeat Customer Behavior")
print(repeat_customers.head())


monthly_trend = df.groupby(['Year', 'Month'])[['Sales', 'Profit', 'Discount']].sum().reset_index()
print("\n11. Monthly Trends in Sales, Profit, and Discount")
print(monthly_trend.head())
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_trend, x='Month', y='Sales', hue='Year', marker='o')
plt.title('Monthly Sales Trend by Year')
plt.tight_layout()
plt.show()

monthly_peak = monthly_trend.sort_values(by='Sales', ascending=False).head(10)
print("\n12. Peak Sales Months")
print(monthly_peak)

value_per_unit = df.groupby('Sub-Category').agg({'Sales': 'sum', 'Quantity': 'sum'})
value_per_unit['Sales per Unit'] = value_per_unit['Sales'] / value_per_unit['Quantity']
value_per_unit = value_per_unit.reset_index()
print("\n13. Sub-Category Sales per Unit")
print(value_per_unit)

market_basket = df.groupby('Order ID')['Product Name'].apply(lambda x: list(x)).reset_index()
print("\n14. Sample Product Combinations")
print(market_basket.head())

X = df[['Sales', 'Quantity', 'Discount']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression().fit(X_train, y_train)
model_score = model.score(X_test, y_test)
print(f"\n15. Predictive Model R² Score for Profit Estimation: {model_score:.4f}")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv("household_power_consumption.csv")

# Drop rows with missing values
data = data.dropna()

# Separate features and target variable
X = data.drop(columns=["Global_active_power"])  # Features
y = data["Global_active_power"]  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: PCA Model Creation
# Initialize PCA model
pca = PCA()

# Fit PCA to scaled data
pca.fit(X_scaled)

# Step 4: Variance Explained
# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# Determine the number of components to explain at least 90% of the variance
explained_variance = 0.90
cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
n_components = (cumulative_variance_ratio < explained_variance).sum() + 1

print(f"Number of components to retain 90% variance: {n_components}")

# Step 5: Dimensionality Reduction
# Retain desired number of principal components
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)


print("Shape of original data:", X.shape)
print("Shape of reduced data:", X_pca.shape)

# Step 6: Model Evaluation
# Split the original data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor on the original data
rf_regressor_orig = RandomForestRegressor(random_state=42)
rf_regressor_orig.fit(X_train, y_train)

# Make predictions on the test set using the original data
y_pred_orig_rf = rf_regressor_orig.predict(X_test)

# Calculate and print the metrics for the original data
accuracy_orig_rf = sum(abs(y_pred_orig_rf - y_test) < 0.5) / len(y_test) * 100
print("Accuracy (Original Data):", accuracy_orig_rf)

print("R-squared (Original Data):", r2_score(y_test, y_pred_orig_rf))


# Split the reduced data into train and test sets
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor on the reduced data
rf_regressor_red = RandomForestRegressor(random_state=42)
rf_regressor_red.fit(X_train_red, y_train_red)

# Make predictions on the test set using the reduced data
y_pred_red_rf = rf_regressor_red.predict(X_test_red)

# Calculate and print the metrics for the reduced data
accuracy_red_rf = sum(abs(y_pred_red_rf - y_test_red) < 0.5) / len(y_test_red) * 100
print("Accuracy (Reduced Data):", accuracy_red_rf)
print("R-squared (Reduced Data):", r2_score(y_test_red, y_pred_red_rf))

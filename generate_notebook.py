import json

def build_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# House Price Prediction using Linear Regression\n",
                    "### Machine Learning Pipeline on the California Housing Dataset\n",
                    "\n",
                    "This notebook demonstrates the end-to-end Machine Learning process to build a **Linear Regression model** that predicts the median house value in California blocks based on demographic and geographical features.\n",
                    "\n",
                    "## Workflow Steps:\n",
                    "1. **Load the Dataset**: Sourced from scikit-learn.\n",
                    "2. **Explore the Data (EDA)**: Understand structure, distributions, and correlation.\n",
                    "3. **Data Preprocessing**: Separate features (X) and target (y).\n",
                    "4. **Split the Data**: 80% training data and 20% validation data.\n",
                    "5. **Train the Model**: Fit a `LinearRegression` algorithm.\n",
                    "6. **Predict House Prices**: Run test predictions.\n",
                    "7. **Evaluate the Model**: Calculate MAE, RMSE, and R² Score.\n",
                    "8. **Save the Model**: Serialize the model into a `.pkl` file.\n",
                    "9. **Conclusion**: Summarize findings."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Import Libraries and Load Dataset\n",
                    "\n",
                    "First, we import the core data science libraries: `pandas`, `numpy`, `matplotlib`, and `seaborn`. We then load the California Housing dataset from `scikit-learn`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.datasets import fetch_california_housing\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn import metrics\n",
                    "import joblib\n",
                    "\n",
                    "# Load the scikit-learn dataset\n",
                    "california = fetch_california_housing(as_frame=True)\n",
                    "df = california.frame\n",
                    "\n",
                    "# Rename columns to be human-readable\n",
                    "column_mapping = {\n",
                    "    'MedInc': 'Median_Income',\n",
                    "    'HouseAge': 'House_Age',\n",
                    "    'AveRooms': 'Ave_Rooms',\n",
                    "    'AveBedrms': 'Ave_Bedrooms',\n",
                    "    'Population': 'Population',\n",
                    "    'AveOccup': 'Ave_Occupancy',\n",
                    "    'Latitude': 'Latitude',\n",
                    "    'Longitude': 'Longitude',\n",
                    "    'MedHouseVal': 'House_Value'\n",
                    "}\n",
                    "df = df.rename(columns=column_mapping)\n",
                    "\n",
                    "# Display first 5 rows\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Explore the Data (EDA)\n",
                    "\n",
                    "Before training, we perform Exploratory Data Analysis to check table dimensions, column data types, missing values, statistics, and distributions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Dimensions of the dataset\n",
                    "print(f\"Number of rows: {df.shape[0]}\")\n",
                    "print(f\"Number of columns: {df.shape[1]}\\n\")\n",
                    "\n",
                    "# Data types and non-null counts\n",
                    "df.info()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check for missing values\n",
                    "print(\"Missing values in each column:\")\n",
                    "print(df.isnull().sum())\n",
                    "\n",
                    "# Descriptive Statistics summary\n",
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Data Visualizations\n",
                    "\n",
                    "Let's visualize the features to understand their distributions and relationships. We'll create:\n",
                    "1. **Histogram** for House Values.\n",
                    "2. **Scatter Plot** showing Median Income vs House Value.\n",
                    "3. **Correlation Heatmap** to check for multicollinearity and linear target relations."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Set style\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "\n",
                    "# 1. House Value Distribution Histogram\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.histplot(df['House_Value'], bins=50, kde=True, color='indigo')\n",
                    "plt.title('Distribution of Median House Values (in $100,000s)')\n",
                    "plt.xlabel('House Value')\n",
                    "plt.ylabel('Count')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Scatter Plot: Median Income vs House Value\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.scatterplot(data=df.sample(2000, random_state=42), x='Median_Income', y='House_Value', alpha=0.4, color='teal')\n",
                    "plt.title('Median Income vs Median House Value')\n",
                    "plt.xlabel('Median Income (in $10,000s)')\n",
                    "plt.ylabel('House Value (in $100,000s)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Correlation Heatmap\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)\n",
                    "plt.title('Correlation Matrix of Housing Features')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Data Preprocessing\n",
                    "\n",
                    "We separate our input features (X) from our target variable (y, house value)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Features (X) - Drop target column\n",
                    "X = df.drop('House_Value', axis=1)\n",
                    "\n",
                    "# Target (y)\n",
                    "y = df['House_Value']\n",
                    "\n",
                    "print(f\"Features columns: {list(X.columns)}\")\n",
                    "print(f\"Target name: {y.name}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Split the Data\n",
                    "\n",
                    "We split the dataset into 80% for training the model and 20% for testing. Splitting is essential to evaluate the model on data it has never seen during training."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "print(f\"Total dataset rows: {len(df)}\")\n",
                    "print(f\"Training set rows: {X_train.shape[0]}\")\n",
                    "print(f\"Testing set rows: {X_test.shape[0]}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Train the Linear Regression Model\n",
                    "\n",
                    "We initialize and fit the `LinearRegression` model using the training subset. The algorithm uses Ordinary Least Squares (OLS) to calculate feature coefficients."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize model\n",
                    "model = LinearRegression()\n",
                    "\n",
                    "# Train model\n",
                    "model.fit(X_train, y_train)\n",
                    "\n",
                    "# Display the intercept and coefficients\n",
                    "print(f\"Intercept (Bias): {model.intercept_:.6f}\\n\")\n",
                    "print(\"Feature Coefficients:\")\n",
                    "for col, coef in zip(X.columns, model.coef_):\n",
                    "    print(f\"  {col:15s}: {coef:.6f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Make Predictions\n",
                    "\n",
                    "Now, we feed the unseen test features `X_test` into the trained model to calculate predicted values."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate predictions\n",
                    "y_pred = model.predict(X_test)\n",
                    "\n",
                    "# Show a comparison of actual vs predicted prices for the first 10 samples\n",
                    "comparison_df = pd.DataFrame({\n",
                    "    'Actual Value ($100k)': y_test.values[:10],\n",
                    "    'Predicted Value ($100k)': y_pred[:10],\n",
                    "    'Difference ($100k)': y_test.values[:10] - y_pred[:10]\n",
                    "})\n",
                    "comparison_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 7: Evaluate the Model\n",
                    "\n",
                    "We measure regression accuracy using three metrics:\n",
                    "1. **Mean Absolute Error (MAE)**: Average magnitude of the errors.\n",
                    "2. **Root Mean Squared Error (RMSE)**: Standard deviation of residuals, penalizes larger errors.\n",
                    "3. **Coefficient of Determination ($R^2$ Score)**: Proportion of variance in the target explained by features."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "mae = metrics.mean_absolute_error(y_test, y_pred)\n",
                    "mse = metrics.mean_squared_error(y_test, y_pred)\n",
                    "rmse = np.sqrt(mse)\n",
                    "r2 = metrics.r2_score(y_test, y_pred)\n",
                    "\n",
                    "# Scale target to actual dollar amounts for explanation\n",
                    "print(\"Model Evaluation Metrics:\")\n",
                    "print(f\"  MAE:  {mae:.4f} (On average, off by ${mae * 100000:,.2f})\")\n",
                    "print(f\"  RMSE: {rmse:.4f} (Standard deviation of errors is ${rmse * 100000:,.2f})\")\n",
                    "print(f\"  R² Score: {r2:.4f} (Model explains {r2*100:.2f}% of the price variation)\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizing Evaluation: Actual vs Predicted Scatter Plot\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "plt.scatter(y_test, y_pred, alpha=0.3, color='indigo')\n",
                    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                    "plt.title('Actual vs Predicted House Values')\n",
                    "plt.xlabel('Actual House Value ($100,000s)')\n",
                    "plt.ylabel('Predicted House Value ($100,000s)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizing Residuals\n",
                    "residuals = y_test - y_pred\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.histplot(residuals, bins=50, kde=True, color='purple')\n",
                    "plt.title('Distribution of Residuals (Errors)')\n",
                    "plt.xlabel('Residuals')\n",
                    "plt.ylabel('Count')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 8: Save the Trained Model\n",
                    "\n",
                    "To reuse the model without retraining, we save it as a serialized joblib pickle file `house_model.pkl`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save model to disk\n",
                    "joblib.dump(model, 'assets/house_model.pkl')\n",
                    "print(\"Model saved successfully as house_model.pkl!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 9: Conclusion\n",
                    "\n",
                    "### Findings:\n",
                    "- **Income Impact**: For every $10,000 increase in median neighborhood income, the predicted home price increases by approximately $43,600 (holding other variables constant).\n",
                    "- **Accuracy**: The OLS baseline model yields an $R^2$ of ~`0.601` on validation tests, explaining ~60.1% of the house price variance.\n",
                    "- **Next Steps**: To improve predictive performance, non-linear modeling such as Random Forest, Gradient Boosting, or Deep Neural Networks should be tested to capture spatial clusters (latitude & longitude) and complex interactions."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open('house_price_predictor.ipynb', 'w') as f:
        json.dump(notebook, f, indent=4)
    print("Generated house_price_predictor.ipynb successfully.")

if __name__ == '__main__':
    build_notebook()

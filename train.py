import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

def main():
    print("Starting House Price Predictor Model Training...")
    
    # 1. Load Dataset
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    # In California Housing dataset:
    # MedInc - median income in block group
    # HouseAge - median house age in block group
    # AveRooms - average number of rooms per household
    # AveBedrms - average number of bedrooms per household
    # Population - block group population
    # AveOccup - average number of household members
    # Latitude - block group latitude
    # Longitude - block group longitude
    # MedHouseVal - median house value in $100,000s (Target)
    
    # Rename columns to be user-friendly for saving csv
    column_mapping = {
        'MedInc': 'Median_Income',
        'HouseAge': 'House_Age',
        'AveRooms': 'Ave_Rooms',
        'AveBedrms': 'Ave_Bedrooms',
        'Population': 'Population',
        'AveOccup': 'Ave_Occupancy',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'MedHouseVal': 'House_Value'
    }
    df_clean = df.rename(columns=column_mapping)
    
    # Ensure assets directory exists
    assets_dir = os.path.join(os.getcwd(), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Save raw/clean dataset CSV
    csv_path = os.path.join(assets_dir, 'california_housing.csv')
    df_clean.to_csv(csv_path, index=False)
    print(f"Saved California Housing dataset to: {csv_path}")
    
    # 2. Separate Features and Target
    X = df_clean.drop('House_Value', axis=1)
    y = df_clean['House_Value']
    
    # 3. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data: Training shape {X_train.shape}, Testing shape {X_test.shape}")
    
    # 4. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression Model trained successfully.")
    
    # 5. Predict on Test Set
    y_pred = model.predict(X_test)
    
    # 6. Evaluate Model
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    
    # Calculate adjusted R2
    n = X_test.shape[0]
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print(f"Evaluation Metrics:")
    print(f"  MAE:  {mae:.4f} (approx ${mae * 100000:.2f})")
    print(f"  RMSE: {rmse:.4f} (approx ${rmse * 100000:.2f})")
    print(f"  R2 Score: {r2:.4f}")
    
    # 7. Save Modelpkl
    pkl_path = os.path.join(assets_dir, 'house_model.pkl')
    joblib.dump(model, pkl_path)
    print(f"Saved joblib model pkl to: {pkl_path}")
    
    # 8. Export Model Parameters and Statistics for Web Application UI
    # Calculate feature stats (min, max, mean, std) to guide UI input validation
    stats = {}
    for col in X.columns:
        stats[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'std': float(X[col].std()),
            'q25': float(X[col].quantile(0.25)),
            'q50': float(X[col].quantile(0.50)),
            'q75': float(X[col].quantile(0.75)),
        }
        
    target_stats = {
        'min': float(y.min() * 100000),
        'max': float(y.max() * 100000),
        'mean': float(y.mean() * 100000),
        'std': float(y.std() * 100000)
    }
    
    params = {
        'coefficients': {col: float(coef) for col, coef in zip(X.columns, model.coef_)},
        'intercept': float(model.intercept_),
        'metrics': {
            'mae': float(mae * 100000), # scale to dollars
            'rmse': float(rmse * 100000), # scale to dollars
            'r2': float(r2),
            'adj_r2': float(adj_r2),
            'total_samples': int(len(df_clean)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        },
        'feature_stats': stats,
        'target_stats': target_stats,
        'sample_predictions': [
            {
                'actual': float(y_test.iloc[i] * 100000),
                'predicted': float(y_pred[i] * 100000),
                'features': {col: float(X_test.iloc[i][col]) for col in X.columns}
            }
            for i in range(100) # Export 100 samples for interactive charts in Web App
        ]
    }
    
    # Save params JSON
    params_path = os.path.join(os.getcwd(), 'model_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Exported model parameters and samples to: {params_path}")
    
    print("Training process finished successfully!")

if __name__ == '__main__':
    main()

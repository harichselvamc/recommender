import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

class MutualFundRecommender:
    def __init__(self, csv_file):
     
        self.df = pd.read_csv(csv_file)
        self.preprocess()
        self.print_unique_risk_levels()

    def preprocess(self):
        """
        Preprocess the data:
          - Convert key columns to numeric.
          - Drop rows missing critical numeric values.
          - For risk_level, if not numeric, force conversion to normalized strings.
        """
        numeric_columns = [
            'min_sip', 'min_lumpsum', 'expense_ratio', 'fund_size_cr',
            'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta', 'sharpe',
            'returns_1yr', 'returns_3yr', 'returns_5yr'
        ]
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df.dropna(subset=numeric_columns, inplace=True)
       
        if 'risk_level' in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df['risk_level']):
                self.df['risk_level'] = self.df['risk_level'].fillna("").astype(str).str.strip().str.lower()

    def print_unique_risk_levels(self):
        """
        Print the available risk levels in the CSV.
        """
        if 'risk_level' in self.df.columns:
            unique_levels = self.df['risk_level'].unique()
            print("Available risk levels in CSV:", unique_levels)

    def cluster_funds(self, filtered_df, horizon, n_clusters=3):
        """
        Apply K-Means clustering on select features:
          - Features: expense_ratio, chosen horizon returns, sharpe, risk_level.
        """
        returns_col = 'returns_1yr' if horizon == 1 else (
                      'returns_3yr' if horizon == 3 else 'returns_5yr')
        features = filtered_df[['expense_ratio', returns_col, 'sharpe', 'risk_level']].copy()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        filtered_df['cluster'] = kmeans.fit_predict(scaled_features)
        return filtered_df

    def compute_composite_score(self, filtered_df, horizon, risk_tolerance):
        """
        Compute a composite score for ranking funds:
          composite_score = 0.5 * (norm_returns)
                          - 0.2 * (norm_expense_ratio)
                          + 0.2 * (norm_sharpe)
                          - 0.1 * (norm_risk_diff)
        where risk_diff = |fund_risk_level - user_risk_tolerance|
        """
        returns_col = 'returns_1yr' if horizon == 1 else (
                      'returns_3yr' if horizon == 3 else 'returns_5yr')
        filtered_df['risk_diff'] = (filtered_df['risk_level'] - risk_tolerance).abs()
        features = filtered_df[[returns_col, 'expense_ratio', 'sharpe', 'risk_diff']].copy()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        weights = np.array([0.5, -0.2, 0.2, -0.1])
        composite_scores = scaled_features.dot(weights)
        filtered_df['composite_score'] = composite_scores
        return filtered_df.sort_values('composite_score', ascending=False)

    def tune_regression_model(self, X, y):
        """
        Use GridSearchCV to tune a Random Forest regressor.
        Returns the best estimator along with cross-validated RMSE.
        """
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        cv_scores = cross_val_score(best_model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-cv_scores)
        print("Best regression parameters:", grid_search.best_params_)
        print("Cross-validated RMSE:", np.mean(rmse_scores))
        return best_model

    def predict_future_return(self, df, horizon):
        """
        Predict future return using a tuned Random Forest regression model.
        The features used include expense_ratio, fund_size_cr, fund_age_yr, sortino,
        alpha, sd, beta, sharpe, and risk_level.
        """
        target = 'returns_1yr' if horizon == 1 else (
                 'returns_3yr' if horizon == 3 else 'returns_5yr')
        features = ['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino',
                    'alpha', 'sd', 'beta', 'sharpe', 'risk_level']
        X = df[features]
        y = df[target]
      
        best_model = self.tune_regression_model(X, y)
       
        df['predicted_return'] = best_model.predict(X)
        
        importances = best_model.feature_importances_
        feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
        print("Feature importances from the regression model:\n", feature_importance)
        return df

    def compute_final_score(self, df, weight_pred=0.3):
        """
        Combine the composite score and the standardized predicted return for final ranking.
          final_score = composite_score + weight_pred * (standardized predicted_return)
        """
        scaler = StandardScaler()
        df['predicted_return_scaled'] = scaler.fit_transform(df[['predicted_return']])
        df['final_score'] = df['composite_score'] + weight_pred * df['predicted_return_scaled']
        return df

    def recommend_funds(self, risk_tolerance, investment_mode, available_amount, horizon, use_ml=True):
        """
        Recommend mutual funds based on the following pipeline:
          1. Filter by risk tolerance and investment mode.
          2. Apply clustering.
          3. Compute a composite score.
          4. Predict future return with a tuned regression model.
          5. Compute a final combined score for ranking.
        """
        filtered_df = self.df.copy()
   
        if 'risk_level' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['risk_level']):
            try:
                risk_value = float(risk_tolerance)
            except ValueError:
                print(f"Risk tolerance input '{risk_tolerance}' cannot be converted to a numeric value.")
                return pd.DataFrame()
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_value]
            if filtered_df.empty:
                print(f"No funds found matching risk tolerance: {risk_tolerance}")
                return filtered_df
        else:
            risk_input = str(risk_tolerance).strip().lower()
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_input]
            if filtered_df.empty:
                print(f"No funds found matching risk tolerance: {risk_tolerance}")
                return filtered_df

 
        mode_input = investment_mode.strip().lower()
        if mode_input == 'sip':
            filtered_df = filtered_df[filtered_df['min_sip'] <= available_amount]
        elif mode_input == 'lumpsum':
            filtered_df = filtered_df[filtered_df['min_lumpsum'] <= available_amount]
        else:
            print("Invalid investment mode. Please choose 'SIP' or 'Lumpsum'.")
            return pd.DataFrame()
        if filtered_df.empty:
            print("No funds meet the minimum investment criteria for the chosen mode.")
            return filtered_df

        if use_ml:
            
            filtered_df = self.cluster_funds(filtered_df, horizon)
            cluster_group = filtered_df.groupby('cluster')['risk_level'].mean()
            print("Average risk level by cluster:\n", cluster_group)
            
            filtered_df = self.compute_composite_score(filtered_df, horizon, risk_tolerance)
         
            filtered_df = self.predict_future_return(filtered_df, horizon)
          
            filtered_df = self.compute_final_score(filtered_df, weight_pred=0.3)
            recommended_df = filtered_df.sort_values('final_score', ascending=False)
        else:
            if horizon == 1:
                returns_col = 'returns_1yr'
            elif horizon == 3:
                returns_col = 'returns_3yr'
            elif horizon == 5:
                returns_col = 'returns_5yr'
            else:
                print("Invalid horizon value. Please choose among: 1, 3, or 5 (years).")
                return pd.DataFrame()
            recommended_df = filtered_df.sort_values(by=returns_col, ascending=False)
        
        return recommended_df


# --- Main Execution ---
if __name__ == '__main__':

    csv_filename = 'comprehensive_mutual_funds_data.csv'
    recommender = MutualFundRecommender(csv_filename)

    risk_tolerance = 3           
    investment_mode = "SIP"      # Options: "SIP" or "Lumpsum"
    available_amount = 1500.0    # The investment amount available
    horizon = 3                  # Investment horizon: 1, 3, or 5 years
    use_ml = True                # Enable ML-based enhancements
    
    print("\nDummy inputs used:")
    print("Risk Tolerance:", risk_tolerance)
    print("Investment Mode:", investment_mode)
    print("Available Amount:", available_amount)
    print("Investment Horizon (years):", horizon)
    print("Using ML-based recommendation:", use_ml)
    print("\nTop mutual fund recommendations based on the dummy inputs:\n")
    
    recommended_funds = recommender.recommend_funds(
        risk_tolerance, investment_mode, available_amount, horizon, use_ml
    )
    
    if not recommended_funds.empty:
        print(recommended_funds.head(10).to_string(index=False))
    else:
        print("No funds match the provided criteria.")

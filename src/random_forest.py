import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint

def run_random_forest(filepath):
    """Train Random Forest with hyperparameter tuning via RandomizedSearchCV."""
    print("\n=== Đang chạy thuật toán: Random Forest ===")
    df = pd.read_csv(filepath)
    # Remove whitespace from columns and values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    categorical_cols = ['model', 'transmission', 'fuelType']
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Random Forest model with tuning parameter grid
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_dist = {
        'regressor__n_estimators': randint(100, 300),
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2']
    }
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf)])
    
    # Hyperparameter optimization
    print("Đang tối ưu tham số (Tuning)...")
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=5, cv=3, 
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_, X_test, y_test
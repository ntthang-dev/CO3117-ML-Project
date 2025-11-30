import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_linear_regression(filepath):
    """Train linear regression baseline model."""
    print("\n=== Đang chạy thuật toán: Linear Regression (Baseline) ===")
    df = pd.read_csv(filepath)
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features and encode categorical features
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['year', 'mileage', 'tax', 'mpg', 'engineSize']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['model', 'transmission', 'fuelType'])
    ])

    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model.fit(X_train, y_train)
    return model, X_test, y_test
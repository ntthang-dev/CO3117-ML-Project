# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_clean_data(filepath):
    """Load data and remove leading/trailing whitespace."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {filepath}")

    # Remove whitespace from column names
    df.columns = df.columns.str.strip()

    # Trim whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    # Optional: Filter outliers using IQR method
    # Q1 = df['price'].quantile(0.25)
    # Q3 = df['price'].quantile(0.75)
    # IQR = Q3 - Q1
    # df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]
    
    return df

def get_preprocessor():
    """Build preprocessing pipeline for numerical and categorical features."""
    numeric_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    categorical_features = ['model', 'transmission', 'fuelType']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop unlisted columns
    )
    
    return preprocessor

def prepare_data(filepath, test_size=0.2, random_state=42):
    """Load, clean, and split data into train/test sets."""
    df = load_and_clean_data(filepath)
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
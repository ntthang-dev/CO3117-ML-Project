import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

def run_svm(filepath):
    print("\n=== Đang chạy thuật toán: Support Vector Machine (SVM) ===")
    df = pd.read_csv(filepath)
    
    # Logic xử lý outliers riêng của nhóm SVM (nếu có)
    # (Giữ nguyên phần tiền xử lý chung để code gọn)
    
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline riêng của SVM (có thêm SelectKBest và Scaler)
    numeric_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    categorical_features = ['model', 'transmission', 'fuelType']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_regression, k='all')) # SVM notebook dùng cái này
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel='rbf'))
    ])

    # GridSearch (rút gọn để chạy nhanh demo, bạn có thể thêm param từ notebook)
    param_grid = {
        'regressor__C': [100, 500], # Lấy giá trị tốt nhất từ notebook
        'regressor__epsilon': [0.1, 0.2],
        'regressor__gamma': ['scale']
    }
    
    print("Đang tối ưu tham số SVM...")
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    
    return search.best_estimator_, X_test, y_test
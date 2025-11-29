import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_ensemble(filepath):
    print("\n=== Đang chạy thuật toán: Voting Ensemble ===")
    df = pd.read_csv(filepath)
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline chung cho Voting (để đơn giản hóa việc kết hợp)
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['year', 'mileage', 'tax', 'mpg', 'engineSize']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['model', 'transmission', 'fuelType'])
    ])

    # Khởi tạo các model con với tham số tốt nhất (giả định đã tìm được)
    estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=120, random_state=42)),
        ('svm', SVR(C=500, kernel='rbf')),
        ('mlp', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42))
    ]

    voting = VotingRegressor(estimators=estimators)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('voting', voting)])
    
    model.fit(X_train, y_train)
    return model, X_test, y_test
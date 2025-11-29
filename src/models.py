# src/models.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def get_linear_regression(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

def get_random_forest(preprocessor, best_params=None):
    # Tham số mặc định hoặc tốt nhất từ notebook
    if best_params is None:
        best_params = {
            'n_estimators': 121, 
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1, **best_params))
    ])

def get_svr(preprocessor):
    # Tham số tốt nhất từ notebook SVM
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(C=500, epsilon=0.2, gamma='scale'))
    ])

def get_mlp(preprocessor):
    # Mạng nơ-ron
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50),
                                   activation='relu',
                                   solver='adam',
                                   max_iter=500,
                                   random_state=42,
                                   early_stopping=True))
    ])

def get_voting_regressor(preprocessor):
    # Kết hợp các mô hình trên
    # Lưu ý: Để VotingRegressor hoạt động trong Pipeline chung, ta cần cẩn thận.
    # Cách tốt nhất là định nghĩa các pipeline con (đã bao gồm preprocessor) làm estimators.
    
    lr = get_linear_regression(preprocessor)
    rf = get_random_forest(preprocessor)
    svr = get_svr(preprocessor)
    mlp = get_mlp(preprocessor)
    
    # VotingRegressor không cần preprocessor riêng vì các model con đã có pipeline riêng
    voting = VotingRegressor(estimators=[
        ('lr', lr),
        ('rf', rf),
        ('svr', svr),
        ('mlp', mlp)
    ])
    
    return voting
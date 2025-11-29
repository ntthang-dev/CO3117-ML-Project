import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import evaluate_model
# Import từng thuật toán riêng biệt
from src.linear_regression import run_linear_regression
from src.random_forest import run_random_forest
from src.svm import run_svm
from src.mlp import run_mlp
from src.ensemble import run_ensemble

def main():
    DATA_PATH = 'data/audi.csv'
    results = []

    # 1. Chạy Linear Regression
    model_lr, X_test, y_test = run_linear_regression(DATA_PATH)
    res_lr = evaluate_model(model_lr, X_test, y_test, "Linear Regression")
    results.append(res_lr)

    # 2. Chạy Random Forest (có Tuning)
    model_rf, X_test_rf, y_test_rf = run_random_forest(DATA_PATH)
    res_rf = evaluate_model(model_rf, X_test_rf, y_test_rf, "Random Forest (Tuned)")
    results.append(res_rf)

    # 3. Chạy SVM (có Tuning)
    model_svm, X_test_svm, y_test_svm = run_svm(DATA_PATH)
    res_svm = evaluate_model(model_svm, X_test_svm, y_test_svm, "SVM (Tuned)")
    results.append(res_svm)
    
    # 4. Chạy MLP (Deep Learning)
    model_mlp, X_test_mlp, y_test_mlp = run_mlp(DATA_PATH)
    res_mlp = evaluate_model(model_mlp, X_test_mlp, y_test_mlp, "MLP (Deep Learning)")
    results.append(res_mlp)

    # 5. Chạy Ensemble Learning
    model_ens, X_test_ens, y_test_ens = run_ensemble(DATA_PATH)
    res_ens = evaluate_model(model_ens, X_test_ens, y_test_ens, "Voting Ensemble")
    results.append(res_ens)

    # Tổng hợp kết quả
    print("\n=== TỔNG KẾT KẾT QUẢ ===")
    df_res = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    print(df_res)

    # Vẽ biểu đồ so sánh R2
    plt.figure(figsize=(12, 6))
    sns.barplot(x='R2', y='Model', data=df_res, palette='viridis')
    plt.title('So sánh độ chính xác (R2 Score) giữa các thuật toán')
    plt.xlim(0.6, 1.0)
    plt.show()

if __name__ == "__main__":
    main()
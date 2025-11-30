import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance and visualize predictions vs actual values."""
    print(f"\n--- Đánh giá: {model_name} ---")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # Scatter plot with perfect fit reference line
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự đoán")
    plt.title(f"{model_name}: Thực tế vs Dự đoán")
    plt.legend()
    plt.show()

    return {'Model': model_name, 'R2': r2, 'RMSE': rmse}
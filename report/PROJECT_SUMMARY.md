# Project Summary: Car Price Prediction

## Team Composition & Contributions

| Role | Member | Contributions |
| :--- | :--- | :--- |
| **Team Lead** | `ntthang-dev` | **Project Rescue & Advanced Analysis**<br>- Implemented **Linear Regression** (Baseline) due to missing module.<br>- Implemented **Deep Learning (MLPRegressor)**.<br>- Performed **Hyperparameter Tuning** for Random Forest.<br>- Built **Ensemble Model** (VotingRegressor).<br>- Integrated all code & fixed compatibility issues. |
| **Member** | `SVM-Dev` | **Support Vector Machine (SVM)**<br>- Initial implementation of SVR model.<br>- Basic EDA and correlation analysis. |
| **Member** | `bao.dang` | **Random Forest**<br>- Initial implementation of Random Forest model. |

## Model Performance Summary

The project successfully implemented and compared 4 distinct models plus an ensemble.

| Model | Status | Performance (R2) | Notes |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | ✅ Implemented by Leader | **Baseline** | Simple, interpretable baseline. |
| **SVM (Optimized)** | ✅ Integrated | **High** | Optimized with `C=500`, `epsilon=0.2`. |
| **Random Forest** | ✅ Tuned by Leader | **Very High** | Tuned `n_estimators`, `max_depth`. |
| **MLP (Deep Learning)** | ✅ Implemented by Leader | **High** | Captures non-linear complex patterns. |
| **Ensemble** | ✅ Implemented by Leader | **Best** | Combines all 4 models for maximum robustness. |

## Conclusion
Despite missing contributions from some team members, the **Team Lead** successfully filled the gaps by implementing the baseline Linear Regression and an advanced Deep Learning model. The final **Ensemble Model** represents the state-of-the-art solution for this dataset, leveraging the diverse strengths of linear, tree-based, kernel-based, and neural network approaches.

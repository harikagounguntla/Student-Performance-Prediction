import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def generate_reports(y_test, y_pred, report_dir):
    """
    Generates all regression evaluation reports
    """
    os.makedirs(report_dir, exist_ok=True)

    # 1️⃣ Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 2️⃣ Save metrics.json
    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 3️⃣ Save evaluation_summary.txt
    summary = f"""
MODEL EVALUATION SUMMARY
------------------------
Generated on : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RMSE : {rmse:.4f}
MAE  : {mae:.4f}
R²   : {r2:.4f}
"""
    with open(os.path.join(report_dir, "evaluation_summary.txt"), "w") as f:
        f.write(summary)

    # 4️⃣ Prediction vs Actual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "prediction_vs_actual.png"))
    plt.close()

    # 5️⃣ Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "residuals_plot.png"))
    plt.close()

    print(f"✅ All reports saved to '{report_dir}/'")

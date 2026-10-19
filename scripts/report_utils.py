import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_metrics(y_true, y_pred, report_dir):
    """
    Saves regression metrics to metrics.json
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }

    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return rmse, mae, r2


def save_evaluation_summary(y_true, y_pred, report_dir):
    """
    Saves human-readable evaluation summary
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

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


def plot_prediction_vs_actual(y_true, y_pred, report_dir):
    """
    Saves Prediction vs Actual plot
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color='red',
        linestyle="--"
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.show() 

    plt.savefig(os.path.join(report_dir, "prediction_vs_actual.png"))
    plt.close()

def plot_residuals(y_true, y_pred, report_dir):
    """
    Saves and displays Residuals plot
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='black', linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.tight_layout()
    
    # ADD THIS LINE to see the plot in a popup window
    plt.show() 
    
    plt.savefig(os.path.join(report_dir, "residuals_plot.png"))
    plt.close()


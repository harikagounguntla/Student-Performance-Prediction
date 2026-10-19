from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ModelEvaluation:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        return {
            "R2 Score": r2,
            "MAE": mae,
            "MSE": mse
        }

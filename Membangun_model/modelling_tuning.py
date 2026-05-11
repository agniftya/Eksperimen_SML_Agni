import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os
import warnings

warnings.filterwarnings('ignore')

dagshub.init(repo_owner='agniftya', repo_name='MLOps-Restaurant-Model', mlflow=True)

mlflow.set_experiment("Restaurant_Satisfaction_Tuning")

def main():
    print("Memuat dataset hasil preprocessing...")
  
    train_path = 'restaurant_satisfaction_preprocessing/train_preprocessed.csv'
    test_path = 'restaurant_satisfaction_preprocessing/test_preprocessed.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Memisahkan Fitur dan Target
    X_train = train_data.drop('HighSatisfaction', axis=1)
    y_train = train_data['HighSatisfaction']
    X_test = test_data.drop('HighSatisfaction', axis=1)
    y_test = test_data['HighSatisfaction']

    # Memulai pencatatan (Tracking) ke MLflow
    with mlflow.start_run():
        print("Memulai Hyperparameter Tuning dengan GridSearchCV...")

        # Algoritma & Parameter yang akan diuji
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        }

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)

        # Prediksi ke data test
        print("Melakukan prediksi dan menghitung metrik...")
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Akurasi Model: {accuracy:.4f}")
        
        print("Membuat visualisasi artefak tambahan...")
        
        # Artefak 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path) 
        plt.close()

        # Artefak 2: Feature Importance
        feature_importances = best_model.feature_importances_
        plt.figure(figsize=(10, 6))
        indices = feature_importances.argsort()[-10:][::-1]
        sns.barplot(x=feature_importances[indices], y=X_train.columns[indices])
        plt.title("Top 10 Feature Importance")
        plt.tight_layout()
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path) 
        plt.close()

        mlflow.sklearn.log_model(best_model, "random_forest_model")

        print("Selesai! Seluruh data eksperimen berhasil dikirim ke DagsHub.")

if __name__ == "__main__":
    main()
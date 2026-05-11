import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

warnings.filterwarnings('ignore')

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing data...")
    
    # 1. Load Data
    data_path = 'restaurant_satisfaction_raw.csv'
    if not os.path.exists(data_path):
        print(f"File {data_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(data_path)
    df_clean = df.copy()

    # 2. Cleaning Data
    df_clean = df_clean.dropna()
    df_clean = df_clean.drop_duplicates()
    df_clean.drop('CustomerID', axis=1, inplace=True)

    # 3. Feature Engineering
    df_clean['SpendPerPerson'] = df_clean['AverageSpend'] / df_clean['GroupSize']

    # 4. Outlier Handling (IQR)
    Q1 = df_clean['AverageSpend'].quantile(0.25)
    Q3 = df_clean['AverageSpend'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    df_clean = df_clean[(df_clean['AverageSpend'] >= lower_bound) & (df_clean['AverageSpend'] <= upper_bound)]

    # 5. Binning
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[0, 25, 59, 100], labels=['Youth', 'Adult', 'Senior'])
    df_clean.drop('Age', axis=1, inplace=True)

    # 6. Splitting & Pipeline
    X = df_clean.drop('HighSatisfaction', axis=1)
    y = df_clean['HighSatisfaction']

    categorical_cols = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit', 'DiningOccasion', 'MealType', 'AgeGroup']
    numerical_cols = ['Income', 'AverageSpend', 'GroupSize', 'WaitTime', 'ServiceRating', 'FoodRating', 'AmbianceRating', 'SpendPerPerson', 'OnlineReservation', 'DeliveryOrder', 'LoyaltyProgramMember']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    preprocessor.set_output(transform='pandas')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 7. Menyimpan Hasil
    train_final = X_train_processed.copy()
    train_final['HighSatisfaction'] = y_train.values 

    test_final = X_test_processed.copy()
    test_final['HighSatisfaction'] = y_test.values

    train_final.to_csv('train_preprocessed.csv', index=False)
    test_final.to_csv('test_preprocessed.csv', index=False)
    
    print("Preprocessing berhasil! File 'train_preprocessed.csv' dan 'test_preprocessed.csv' telah dibuat.")

if __name__ == "__main__":
    run_preprocessing()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def main():
    print("Loading data...")
    # Load dataset with correct encoding
    df = pd.read_csv('bi.csv', encoding='latin1')
    
    # 1. Basic Cleaning
    # Drop names as they are not predictive
    df = df.drop(columns=['fNAME', 'lNAME'])
    
    # Standardize text columns
    text_cols = ['gender', 'country', 'residence', 'prevEducation']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            
    # Simplify gender manually due to typos
    df['gender'] = df['gender'].replace({'m': 'male', 'f': 'female'})
    # Simplify country
    df['country'] = df['country'].replace({'norge': 'norway', 'rsa': 'south africa'})
    # Simplify residence
    df['residence'] = df['residence'].replace({'bi-residence': 'bi residence', 'biresidence': 'bi residence'})
    # Simplify prevEducation
    df['prevEducation'] = df['prevEducation'].replace({'highschool': 'high school', 'barrrchelors': 'bachelors'})

    # 2. Handle missing target values
    # Drop rows where 'Python' score is missing
    df = df.dropna(subset=['Python'])
    
    X = df.drop(columns=['Python'])
    y = df['Python']

    # 3. Preprocessing Pipeline
    numeric_features = ['Age', 'entryEXAM', 'studyHOURS', 'DB']
    categorical_features = ['gender', 'country', 'residence', 'prevEducation']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train Model
    print("Training model...")
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model trained. Evaluation:\nMAE: {mae:.2f}\nR2 Score: {r2:.2f}")

    # 8. Save Model
    joblib.dump(model, 'model.pkl')
    print("Model saved to 'model.pkl'")

if __name__ == "__main__":
    main()

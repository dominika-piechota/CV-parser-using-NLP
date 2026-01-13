import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent 
sys.path.append(str(PROJECT_ROOT))

INPUT_FILE = PROJECT_ROOT / "database" / "resumes_dataset_CLEANED.jsonl"
MODEL_OUTPUT_FILE = PROJECT_ROOT / "models" / "best_model_balanced.pkl"

DEGREE_RANKING = {
    "UNKNOWN": 0, "HIGH SCHOOL": 1, "DIPLOMA": 1, "ASSOCIATE": 2,
    "BACHELOR": 3, "BACHELOR'S": 3, "BS": 3, "B.S": 3, "BA": 3, "BSC": 3, "ENGINEER": 3,
    "MASTER": 4, "MASTER'S": 4, "MS": 4, "M.S": 4, "MSC": 4, "MA": 4, "MBA": 5,
    "PHD": 5, "PH.D": 5, "DOCTORATE": 5
}

def to_dense(X): return X.toarray()

def get_degree_rank(d):
    if not d: return 0
    d_up = d.upper()
    best = 0
    for k, v in DEGREE_RANKING.items():
        if k in d_up and v > best: best = v
    return best

def flatten_data(input_path):
    print(f"Data loading: {input_path}...")
    data = []
    if not input_path.exists():
        print("ERROR: No input file found")
        return pd.DataFrame()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                feat = rec.get("features", {})
                
                max_rank = 0
                fields = []
                for edu in feat.get("education", []):
                    r = get_degree_rank(edu.get("degree"))
                    if r > max_rank: max_rank = r
                    fld = edu.get("field_of_study")
                    if fld and fld != "Unknown": fields.append(fld)
                
                skills = feat.get("skills", [])
                skills_str = " ".join(skills) if isinstance(skills, list) else str(skills)

                data.append({
                    "score": float(rec.get("score", 0)),
                    "target_role": rec.get("target_role", "Unknown"),
                    "summary": feat.get("summary_clean", ""),
                    "skills_text": skills_str,
                    "degree_rank": max_rank,
                    "fields_text": " ".join(fields)
                })
            except: continue
    return pd.DataFrame(data)

def run_optimization_and_training():
    df_raw = flatten_data(INPUT_FILE)
    if df_raw.empty: return

    print(f"\n[0/4] Data Balancing & Preprocessing")
    print(f"   Original dataset size: {len(df_raw)}")
    
    df_high = df_raw[df_raw['score'] >= 3.5] # Good matches
    df_low = df_raw[df_raw['score'] < 3.5] # Poor matches
    
    print(f"   High scores (>=3.5): {len(df_high)}")
    print(f"   Low scores (<3.5):   {len(df_low)}")
    
    # Taking only as many low-score samples as 1.5x the high-score samples (natural imbalance)
    if len(df_low) > 1.5 * len(df_high):
        df_low_sampled = df_low.sample(n=int(1.5 * len(df_high)), random_state=42)
        df = pd.concat([df_high, df_low_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   Balanced dataset size: {len(df)}")
    else:
        df = df_raw
        print("   Dataset is already balanced enough.")
    
    X = df[["target_role", "summary", "skills_text", "degree_rank", "fields_text"]]
    y = df["score"]

    y_stratify = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_stratify
    )
    
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('role', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['target_role']),
            ('summary', TfidfVectorizer(max_features=500, stop_words='english'), 'summary'),
            ('skills', TfidfVectorizer(max_features=300, binary=True), 'skills_text'),
            ('degree', StandardScaler(), ['degree_rank']),
            ('fields', TfidfVectorizer(max_features=50), 'fields_text')
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('dense', FunctionTransformer(to_dense, accept_sparse=True)), 
        ('reg', HistGradientBoostingRegressor(random_state=42))
    ])

    print("\n[1/4] Start Grid Search (on Balanced Data)")
    
    param_grid = {
        'reg__learning_rate': [0.05, 0.1, 0.2], 
        'reg__max_iter': [100, 200, 300], 
        'reg__max_depth': [5, 10, 20],
        'reg__l2_regularization': [0.3, 0.5, 0.7, 1.0]
    }
    
    # Bezpieczne n_jobs=1 dla Windowsa
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2
    )
    
    grid_search.fit(X_train, y_train)

    print(f"\n Best CV params found (R2: {grid_search.best_score_:.4f}):")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # --- EWALUACJA ---
    print("\n[2/4] Evaluating on Test Set")
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("-" * 40)
    print(f"Model Performance Metrics (Test Set):")
    print(f"   R2 Score:         {r2:.4f}")
    print(f"   MAE:              {mae:.4f}")
    print(f"   RMSE:             {rmse:.4f}")
    print("-" * 40)

    # --- RETRAINING ---
    print("\n[3/4] Retraining best model on FULL BALANCED dataset...")
    
    final_model = grid_search.best_estimator_
    final_model.fit(X, y)
    
    MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_OUTPUT_FILE)
    print(f"\n💾 Final Model saved to: {MODEL_OUTPUT_FILE}")

if __name__ == "__main__":
    run_optimization_and_training()
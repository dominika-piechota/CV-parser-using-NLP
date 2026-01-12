import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

INPUT_FILE = Path("./database/resumes_dataset_CLEANED.jsonl")
MODEL_OUTPUT_FILE = Path("./models/best_model_balanced.pkl") 

# Map changing degree strings to ranks (Ordinal Encoding)
DEGREE_RANKING = {
    "UNKNOWN": 0, "HIGH SCHOOL": 1, "DIPLOMA": 1, "ASSOCIATE": 2,
    "BACHELOR": 3, "BACHELOR'S": 3, "BS": 3, "B.S": 3, "BA": 3, "BSC": 3, "ENGINEER": 3,
    "MASTER": 4, "MASTER'S": 4, "MS": 4, "M.S": 4, "MSC": 4, "MA": 4, "MBA": 5,
    "PHD": 5, "PH.D": 5, "DOCTORATE": 5
}

# Conversion function: sparse matrix (from TF-IDF) to dense array needed for HistGradientBoosting
def to_dense(X): return X.toarray()

# Return rank for a given degree string
def get_degree_rank(d):
    if not d: return 0
    d_up = d.upper()
    best = 0
    for k, v in DEGREE_RANKING.items():
        if k in d_up and v > best: best = v
    return best

# Flatten JSONL data into DataFrame suitable for ML
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
                
                # Get the highest degree rank and fields of study
                max_rank = 0
                fields = []
                for edu in feat.get("education", []):
                    r = get_degree_rank(edu.get("degree"))
                    if r > max_rank: max_rank = r
                    fld = edu.get("field_of_study")
                    if fld and fld != "Unknown": fields.append(fld)
                
                # Combine skills into a single string for TF-IDF
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
    df = flatten_data(INPUT_FILE)
    if df.empty: return

    # WEIGHT CALCULATION FOR BALANCING THE DATASET
    # Problem: The dataset contains many low ratings (1.0) and few high ratings (5.0)
    # Solution: We assign weights to the sample. The weight increases exponentially with the rating
    # A record with a score of 5.0 has a weight of ~55, and a record with a score of 1.0 has a weight of 1
    # This forces the model to ‘fight’ for the correct prediction of ideal candidates
    sample_weights = df['score'] ** 2.5
    
    X = df[["target_role", "summary", "skills_text", "degree_rank", "fields_text"]]
    y = df["score"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('role', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['target_role']), # role is categorical
            ('summary', TfidfVectorizer(max_features=2000, stop_words='english'), 'summary'), # text field (key words analysis)
            ('skills', TfidfVectorizer(max_features=800, binary=True), 'skills_text'), # binary TF-IDF for skills (can be present or not)
            ('degree', StandardScaler(), ['degree_rank']), # numerical field (standard scaling)
            ('fields', TfidfVectorizer(max_features=150), 'fields_text')
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('dense', FunctionTransformer(to_dense, accept_sparse=True)), 
        ('reg', HistGradientBoostingRegressor(random_state=42))
    ])

    # Grid Search for hyperparameter tuning with sample weights
    print("\n[1/2] Start Grid Search")
    
    param_grid = {
        'reg__learning_rate': [0.05, 0.1, 0.15, 0.3],
        'reg__max_iter': [200, 300, 400],
        'reg__max_depth': [None, 10, 15, 20],
        'reg__l2_regularization': [0.0, 0.1]
    }

    n_combs = 1
    for k, v in param_grid.items(): n_combs *= len(v)
    print(f"To check: {n_combs} combinations * 3 folds = {n_combs * 3} training runs.")
    
    fit_params = {
        'reg__sample_weight': sample_weights
    }

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='r2', # we want to maximize R2, the coefficient of determination
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X, y, **fit_params)

    print(f"\n Best params found (R2: {grid_search.best_score_:.4f}):")
    print(grid_search.best_params_)

    print("\n[2/2] Saving the best model")
    
    best_model = grid_search.best_estimator_
    
    MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_FILE)
    print(f"Model saved to: {MODEL_OUTPUT_FILE}")
    print("You can now run the script 'predict_career.py'.")

if __name__ == "__main__":
    run_optimization_and_training()
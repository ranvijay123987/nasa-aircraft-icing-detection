# =============================================================================
# AIRCRAFT ICING DETECTION – NASA IRT ONLY (MINIMAL CLEAN PIPELINE)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("====== NASA IRT – ICING DETECTION MODEL ======")

# =============================================================================
# STEP 1: LOAD ONLY NASA IRT DATA
# =============================================================================

def load_nasa_data():
    print("\n[1] Loading NASA IRT CSV...")

    df = pd.read_csv("data/nasa_irt_data.csv")

    print("✓ Loaded:", df.shape, "records")

    # If icing column already exists, return
    if "icing_event" in df.columns:
        return df

    # Auto detect temperature & LWC columns
    temp_col = None
    lwc_col = None

    numeric_df = df.select_dtypes(include=["float64", "int64"])

    for col in numeric_df.columns:
        col_min = numeric_df[col].min()
        col_max = numeric_df[col].max()

        # Temperature heuristic
        if -80 < col_min < 0 and 0 < col_max < 80:
            if temp_col is None:
                temp_col = col

        # LWC heuristic
        if 0 <= col_min <= 5 and 0 < col_max <= 10:
            if lwc_col is None:
                lwc_col = col

    print(f"Auto-detected TEMP column → {temp_col}")
    print(f"Auto-detected LWC column → {lwc_col}")

    if temp_col is None or lwc_col is None:
        raise ValueError(
            "Could not detect temperature or LWC columns.\n"
            "Send me the first 20 rows so I can identify them manually."
        )

    # Create icing event
    df["icing_event"] = (
        (df[temp_col] > -20) &
        (df[temp_col] < 0) &
        (df[lwc_col] > 0.5)
    ).astype(int)

    print("✓ icing_event created using automatically detected columns")

    return df


# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================

def preprocess(df):
    print("\n[2] Preprocessing data...")

    # Separate target if exists
    if "icing_event" in df.columns:
        y = df["icing_event"]
        df = df.drop("icing_event", axis=1)
    else:
        raise ValueError("icing_event column missing after load_nasa_data()")

    # Keep ONLY numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    dropped = set(df.columns) - set(numeric_df.columns)

    print("Dropped non-numeric columns:", dropped)

    X = numeric_df.copy()

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("✓ Preprocessing complete")
    return X, y, scaler, imputer


# =============================================================================
# STEP 3: SPLIT DATA
# =============================================================================

def split_data(X, y):
    print("\n[3] Splitting NASA dataset...")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# STEP 4: TRAIN SIMPLE RANDOM FOREST (NO DNN, NO TUNING)
# =============================================================================

def train_model(X_train, y_train):
    print("\n[4] Training Random Forest...")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("✓ Model trained")
    return model

# =============================================================================
# STEP 5: EVALUATE MODEL
# =============================================================================

def evaluate(model, X_test, y_test):
    print("\n[5] Evaluating model...")

    y_pred = model.predict(X_test)

    print("\n=== TEST RESULTS ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

# =============================================================================
# STEP 6: SAVE MODEL + SCALER
# =============================================================================

def save(model, scaler, imputer):
    print("\n[6] Saving model...")

    joblib.dump(model, "nasa_icing_rf.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(imputer, "imputer.pkl")

    print("✓ Saved nasa_icing_rf.pkl")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    df = load_nasa_data()
    X, y, scaler, imputer = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save(model, scaler, imputer)

    print("\n=== NASA-ONLY PIPELINE COMPLETED ===")

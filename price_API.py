from flask import Flask, request, jsonify, send_from_directory, abort
import pandas as pd
import numpy as np
import pickle
import json
import joblib
from typing import Optional

app = Flask(__name__)


# ===============================
#     Request / Response Model
# ===============================



# ===============================
#        Utilities
# ===============================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def convert_sqft_to_num(x):
    if isinstance(x, (int, float)):
        return x
    try:
        tokens = x.split("-")
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
    except:
        pass
    try:
        return float(x)
    except ValueError:
        if "Sq. Meter" in x:
            return float(x.replace("Sq. Meter", "").strip()) * 10.7639
        if "Acres" in x:
            return float(x.replace("Acres", "").strip()) * 43560
        if "Sq. Yards" in x:
            return float(x.replace("Sq. Yards", "").strip()) * 9
        if "Cents" in x:
            return float(x.replace("Cents", "").strip()) * 435.6
        if "Guntha" in x:
            return float(x.replace("Guntha", "").strip()) * 1089
        if "Perch" in x:
            return float(x.replace("Perch", "").strip()) * 272.25
        if "Grounds" in x:
            return float(x.replace("Grounds", "").strip()) * 2400
        import re
        numeric_part = re.findall(r"\d+\.?\d*", x)
        if numeric_part:
            return float(numeric_part[0])
    return np.nan


# ===============================
#     Artifact Loading
# ===============================
try:
    model = load_pickle("best_random_forest_model.pkl")
    location_price_map = load_pickle("location_price_map.pkl")
    overall_mean_price = load_pickle("overall_mean_price.pkl")
    # Load scaler with fallbacks: scaler.joblib -> scaler.pkl (joblib) -> scaler.pkl (pickle)
    try:
        scaler = joblib.load("scaler.joblib")
    except Exception:
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception:
            scaler = load_pickle("scaler.pkl")

    try:
        with open("feature_columns.json", "r") as f:
            feature_columns = json.load(f)
    except Exception:
        # Fallback: derive from model if trained with DataFrame (scikit-learn >=1.0)
        feature_columns = list(getattr(model, "feature_names_in_", []))
        if not feature_columns:
            raise RuntimeError("Missing feature_columns.json and model lacks feature_names_in_. Export feature_columns.json from training.")
        # Persist for next runs
        with open("feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(feature_columns, f)

    # Optional medians with safe defaults
    try:
        bath_median = load_pickle("bath_median.pkl")
    except Exception:
        bath_median = 2.0
    try:
        balcony_median = load_pickle("balcony_median.pkl")
    except Exception:
        balcony_median = 1.0

except Exception as e:
    raise RuntimeError(f"❌ Missing artifacts: {e}")


# ===============================
#              API
# ===============================
@app.route("/")
def root():
    return jsonify({"message": "House Price API OK", "ui": "/ui"})

@app.route("/ui")
def ui():
    return send_from_directory("templates", "index.html", mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        abort(400, description="Invalid JSON body")
    required = ["location", "total_sqft", "size"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        abort(400, description=f"Missing fields: {', '.join(missing)}")

    # === Build DF ===
    df = pd.DataFrame([{
        "location": data.get("location"),
        "total_sqft": data.get("total_sqft"),
        "bath": data.get("bath"),
        "balcony": data.get("balcony"),
        "size": data.get("size")
    }])

    # === Imputation ===
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce").fillna(bath_median)
    df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce").fillna(balcony_median)

    # === Clean + convert sqft ===
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
    if df["total_sqft"].isna().any():
        abort(422, description="Invalid total_sqft format")

    # === Target encode location ===
    df["location"] = df["location"].str.replace(" ", "").str.lower()
    df["location"] = df["location"].map(location_price_map).fillna(overall_mean_price)

    # === Size conversion ===
    df["size"] = df["size"].astype(str).str.strip().str.lower()
    df["size_num"] = df["size"].str.extract(r"(\d+)").astype(float).fillna(0)
    df["size_type"] = df["size"].str.extract(r"(bhk|bedroom|rk)")

    df["BHK"] = df.apply(lambda x: x["size_num"] if x["size_type"] == "bhk" else 0, axis=1)
    df["Bedroom"] = df.apply(lambda x: x["size_num"] if x["size_type"] == "bedroom" else 0, axis=1)
    df["RK"] = df.apply(lambda x: x["size_num"] if x["size_type"] == "rk" else 0, axis=1)

    # === One-hot ===
    dummies = pd.get_dummies(df["size_type"], prefix="type")

    # Force missing dummies to exist
    for col in ["type_bedroom", "type_bhk", "type_rk"]:
        if col not in dummies.columns:
            dummies[col] = 0

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(["size_num", "size", "size_type"], axis=1)

    # === scaling ===
    df[["location", "total_sqft"]] = scaler.transform(df[["location", "total_sqft"]])

    # === Final column ordering ===
    missing_cols = [c for c in feature_columns if c not in df.columns]
    for col in missing_cols:
        df[col] = 0

    df = df.reindex(columns=feature_columns, fill_value=0)

    # === Prediction ===
    prediction = model.predict(df)[0]
    return jsonify({"predicted_price": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

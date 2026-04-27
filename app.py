from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Carbon_Footprints_Prediction.pkl")
model = joblib.load(MODEL_PATH)

# Fuel type encoding (from notebook)
FUEL_MAPPING = {
    "Natural Gas": 1,
    "Coal": 3,
    "Renewables": 0,
    "Diesel": 2
}

# Fuel type emission context (for display)
FUEL_LABELS = {
    "Natural Gas": {"label": "Natural Gas", "color": "#4a8fa8", "icon": "🔵"},
    "Coal":        {"label": "Coal",         "color": "#6b5b45", "icon": "⚫"},
    "Renewables":  {"label": "Renewables",   "color": "#6b8c3e", "icon": "🟢"},
    "Diesel":      {"label": "Diesel",       "color": "#c85a2a", "icon": "🟠"},
}

def get_rating(value_kg):
    """Rate the carbon footprint level."""
    if value_kg < 8000:
        return {"level": "Low",      "color": "#6b8c3e", "bg": "#edf5d8", "icon": "🌿", "desc": "Well below average — excellent efficiency!"}
    elif value_kg < 15000:
        return {"level": "Moderate", "color": "#4a8fa8", "bg": "#dbeef6", "icon": "💧", "desc": "Below the dataset average. Keep optimising."}
    elif value_kg < 25000:
        return {"level": "Average",  "color": "#e8a020", "bg": "#fdf0d5", "icon": "⚠️", "desc": "Around the industry average. Room to improve."}
    elif value_kg < 35000:
        return {"level": "High",     "color": "#c85a2a", "bg": "#fde8df", "icon": "🔶", "desc": "Above average. Consider efficiency upgrades."}
    else:
        return {"level": "Critical", "color": "#a02020", "bg": "#fde0d8", "icon": "🔴", "desc": "Very high emissions. Urgent action recommended."}

def generate_tips(fuel_type, efficiency, electricity, production):
    tips = []
    if fuel_type == "Coal":
        tips.append("Switch from coal to natural gas or renewables — can cut emissions by 40–60%.")
    if fuel_type == "Diesel":
        tips.append("Consider transitioning to cleaner fuel sources like natural gas or renewables.")
    if efficiency < 0.7:
        tips.append("Improving efficiency rating from your current level to above 0.85 could significantly cut emissions.")
    if electricity > 35000:
        tips.append("Your electricity usage is high — invest in energy-efficient equipment or solar panels.")
    if fuel_type == "Renewables":
        tips.append("Great choice on renewables! Explore battery storage to maximise self-generated energy.")
    if efficiency >= 0.9:
        tips.append("High efficiency rating — maintain it through regular equipment maintenance.")
    tips.append("Track monthly emissions and set reduction targets aligned with industry benchmarks.")
    return tips[:3]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        electricity  = float(data.get("electricity", 0))
        fuel_type    = data.get("fuel_type", "Natural Gas")
        production   = float(data.get("production", 0))
        efficiency   = float(data.get("efficiency", 0.8))

        fuel_encoded = FUEL_MAPPING.get(fuel_type, 1)

        input_df = pd.DataFrame([{
            "Electricity_Usage_kWh":   electricity,
            "Fuel_Type":               fuel_encoded,
            "Production_Volume_Tons":  production,
            "Efficiency_Rating":       efficiency
        }])

        prediction_kg = float(model.predict(input_df)[0])
        prediction_kg = max(0, round(prediction_kg, 1))
        prediction_ton = round(prediction_kg / 1000, 2)

        rating  = get_rating(prediction_kg)
        tips    = generate_tips(fuel_type, efficiency, electricity, production)

        # Dataset stats for comparison
        dataset_avg_kg   = 20852.8
        dataset_min_kg   = 1146.7
        dataset_max_kg   = 44513.0

        # Breakdown estimate (contribution weights from model coefficients)
        elec_contrib  = round(abs(0.505 * electricity), 1)
        fuel_contrib  = round(abs(3504.7 * fuel_encoded), 1)
        prod_contrib  = round(abs(1.789 * production), 1)
        eff_contrib   = round(abs(754.5 * efficiency), 1)
        total_contrib = elec_contrib + fuel_contrib + prod_contrib + eff_contrib

        breakdown = {
            "Electricity":  round(elec_contrib / total_contrib * 100, 1) if total_contrib else 0,
            "Fuel Type":    round(fuel_contrib  / total_contrib * 100, 1) if total_contrib else 0,
            "Production":   round(prod_contrib  / total_contrib * 100, 1) if total_contrib else 0,
            "Efficiency":   round(eff_contrib   / total_contrib * 100, 1) if total_contrib else 0,
        }

        return jsonify({
            "success":        True,
            "prediction_kg":  prediction_kg,
            "prediction_ton": prediction_ton,
            "rating":         rating,
            "tips":           tips,
            "breakdown":      breakdown,
            "dataset_avg_kg": dataset_avg_kg,
            "pct_vs_avg":     round((prediction_kg - dataset_avg_kg) / dataset_avg_kg * 100, 1),
            "fuel_info":      FUEL_LABELS.get(fuel_type, {}),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)

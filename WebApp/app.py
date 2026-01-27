from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np
  # no GUI, safe for servers

from eda_auto import run_eda

app = Flask(__name__)

# Load model
model = xgb.Booster()
model.load_model("xgboost_model.json")

# Folders (relative to WebApp/)
UPLOAD_FOLDER = "uploads"
GRAPH_FOLDER = "static/auto_graphs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# ---------- MAIN DATASET EDA ----------
@app.route('/analysis')
def analysis():
    csv_path = "../data/Cleandata.csv"   # go out of WebApp
    run_eda(csv_path, GRAPH_FOLDER)
    return render_template("analysis.html")

# ---------- UPLOAD PAGE ----------
@app.route('/upload')
def upload():
    return render_template('upload.html')

# ---------- PROCESS UPLOADED FILE ----------
@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path)

    required_cols = ['Severity','Weather_Condition','Visibility(mi)','Street','State']
    for col in required_cols:
        if col not in df.columns:
            return f"Missing required column: {col}"
        
    # After loading df in /process

    insights = []

    # Severity insight
    sev_counts = df['Severity'].value_counts(normalize=True)*100
    top_sev = sev_counts.idxmax()
    insights.append(f"Most accidents fall under Severity {top_sev} ({sev_counts[top_sev]:.1f}%).")

    # Weather insight
    top_weather = df['Weather_Condition'].value_counts().idxmax()
    insights.append(f"Most accidents happened during {top_weather} weather.")

    # Visibility insight
    low_vis = df[df['Visibility(mi)'] < 3]
    if len(low_vis) > 0:
        insights.append("Low visibility (<3 miles) is strongly linked with higher accident severity.")

    # Street insight
    top_street = df['Street'].value_counts().idxmax()
    insights.append(f"Street with highest accident frequency: {top_street}.")

    # Save insights to send UI
    return render_template("auto_analysis.html", insights=insights)


    # Severity
    plt.figure()
    sns.countplot(x='Severity', data=df)
    plt.title("Severity Distribution")
    plt.savefig(f"{GRAPH_FOLDER}/severity.png")
    plt.close()

    # Weather
    plt.figure(figsize=(8,4))
    sns.countplot(x='Weather_Condition', data=df,
                  order=df['Weather_Condition'].value_counts().index[:5])
    plt.xticks(rotation=45)
    plt.title("Top Weather Conditions")
    plt.savefig(f"{GRAPH_FOLDER}/weather.png")
    plt.close()

    # Visibility
    plt.figure()
    sns.boxplot(x='Severity', y='Visibility(mi)', data=df)
    plt.title("Visibility vs Severity")
    plt.savefig(f"{GRAPH_FOLDER}/visibility.png")
    plt.close()

    return render_template("auto_analysis.html")

# ---------- PREDICTION ----------
@app.route('/predict', methods=['POST'])
def predict():
    import numpy as np
    import xgboost as xgb

    # Build input dataframe
    input_data = pd.DataFrame({
        'Year': [2020],
        'Start_Lat': [40.0],
        'Start_Lng': [-82.9],
        'Distance(mi)': [float(request.form['distance'])],
        'Street': [request.form['street']],
        'City': [request.form['city']],
        'County': ['Unknown'],
        'State': [request.form['state']],
        'Airport_Code': ['KCMH'],
        'Temperature(F)': [float(request.form['temperature'])],
        'Wind_Chill(F)': [float(request.form['wind_chill'])],
        'Visibility(mi)': [float(request.form['visibility'])],
        'Wind_Direction': [request.form['wind_direction']],
        'Weather_Condition': [request.form['weather_condition']],
        'Traffic_Signal': [int(request.form['traffic_signal'])],
        'Sunrise_Sunset': [request.form['sunrise_sunset']],
        'TimeDiff': [30]
    })

    # Categorical columns
    cat_cols = [
        'Street', 'City', 'County', 'State',
        'Airport_Code', 'Wind_Direction',
        'Weather_Condition', 'Sunrise_Sunset'
    ]

    for col in cat_cols:
        input_data[col] = input_data[col].astype('category')

    # ✅ Create DMatrix correctly
    dmatrix = xgb.DMatrix(input_data, enable_categorical=True)

    # ✅ Predict (returns probabilities)
    probs = model.predict(dmatrix)

    # ✅ Convert probabilities → class
    prediction = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs))

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=round(confidence, 2)
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)


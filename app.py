from flask import Flask, jsonify, request, send_file, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Ensure there is a folder named 'static' in the same directory as your script
if not os.path.exists('static'):
    os.makedirs('static')

# Load data and create model
url = 'https://drive.google.com/uc?id=1rrNikVGWYxUiDvryTygEJAeU0Z5QF2np'
df = pd.read_csv(url)
predictors = df.drop("target", axis=1)
target = df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

@app.route('/api/predict', methods=['GET'])
def predict():
    new_data = [request.args.get(f'p{i+1}', default=0, type=float) for i in range(13)]
    new_data_scaled = scaler.transform(np.array([new_data]))
    prediction = model.predict(new_data_scaled)[0]
    prediction_text = 'Heart Disease' if prediction == 1 else 'No Heart Disease'

    # LIME Explanation
    column_names = predictors.columns.tolist()
    explainer = LimeTabularExplainer(X_train_scaled, feature_names=column_names,
                                     class_names=['No Heart Disease', 'Heart Disease'], mode='classification')
    exp = explainer.explain_instance(new_data_scaled[0], model.predict_proba, num_features=13, top_labels=1)
    fig, axes = plt.subplots(figsize=(12, 8))
    exp.as_pyplot_figure(label=exp.available_labels()[0])
    plt.title('Predictive Factors for Heart Disease')
    plt.tight_layout()
    image_path = 'static/lime_output.png'  # Save in the static folder
    plt.savefig(image_path, dpi=300)
    plt.close(fig)

    # Create HTML content to display the prediction and the image
    html_content = f"<html><body><h1>Prediction: {prediction_text}</h1>"
    html_content += f"<img src='{url_for('static', filename='lime_output.png')}' alt='LIME Explanation'>"
    html_content += "</body></html>"

    return html_content

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

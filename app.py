import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for
from flask_assets import Bundle, Environment
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_PATH"] = "models/random_forest_model.pkl"
app.config["TEST_ACCURACY_THRESHOLD"] = 0.9586529413442071

# Initialize Flask-Assets
assets = Environment(app)
css = Bundle("src/main.css", output="dist/main.css")

assets.register("css", css)
css.build()

# Load the pre-trained model
model = joblib.load(app.config["MODEL_PATH"])


def preprocess_data(data):
    # Perform necessary preprocessing on the data
    data = data.drop(columns=['id', 'attack_cat'])
    X = data.drop(columns=['label'])  # Features
    y = data['label']  # Target variable
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has a file part
        if "file" not in request.files:
            return render_template("error.html", message="No file part")

        file = request.files["file"]

        if file.filename == "":
            return render_template("error.html", message="No selected file")

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Read the CSV data into a DataFrame
            data_to_classify = pd.read_csv(file_path)

            # Preprocess the data
            X_test, y_test = preprocess_data(data_to_classify)

            # Use the model to make predictions
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            if accuracy <= app.config["TEST_ACCURACY_THRESHOLD"]:
                prediction_message = "Normal"
            else:
                prediction_message = "Anomaly"

            return render_template("result.html", prediction=prediction_message)

    return render_template("index.html")


if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)

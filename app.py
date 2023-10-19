import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for
from flask_assets import Bundle, Environment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_PATH"] = "models/random_forest_model.pkl"
app.config["TEST_ACCURACY_THRESHOLD"] = 0.95

# Initialize Flask-Assets
assets = Environment(app)
css = Bundle("src/main.css", output="dist/main.css")

assets.register("css", css)
css.build()


def make_prediction(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    return (accuracy := accuracy_score(y_test, y_pred))


def preprocess_data(data, model_choice):
    if model_choice == 1:
        # Drop unwanted columns
        data = data.drop(columns=['version', 'ihl', 'tos', 'len', 'id', 'frag', 'ttl', 'chksum',
                                  'src', 'dst', 'options', 'time', 'dataofs', 'reserved',
                                  'window', 'chksum.1', 'urgptr', 'options.1', 'payload',
                                  'payload_raw', 'payload_hex'])

        # Split the data into features (X) and target variable (y)
        X = data.drop(columns=['label'])
        y = data['label']

        # Perform one-hot encoding for categorical columns
        X = pd.get_dummies(X)

        # Normalize or scale features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y
    elif model_choice == 0:
        # Drop unwanted columns
        data = data.drop(columns=['id', 'attack_cat'])

        # Remove rows with missing values
        data = data.dropna()

        # Split the data into features (X) and target variable (y)
        X = data.drop(columns=['label'])
        y = data['label']

        # Perform one-hot encoding for categorical columns
        X = pd.get_dummies(X)

        # Normalize or scale features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y


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

            model_type = int(request.form.get("model_type"))

            # Preprocess the data
            X, y = preprocess_data(data_to_classify, model_type)
            accuracy = make_prediction(X, y)

            if accuracy < app.config["TEST_ACCURACY_THRESHOLD"]:
                prediction_message = "Normal"
            else:
                prediction_message = "Anomaly"

            return render_template("result.html", prediction=prediction_message)

    return render_template("index.html")


if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)

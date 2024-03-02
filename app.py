from flask import Flask, request, render_template
import pickle
import numpy as np
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load your trained XGBoost model
with open("xgb_model_pca_tuned.pkl", "rb") as file:
    xgb_model = pickle.load(file)
with open("rf_model_pca_tuned.pkl", "rb") as file:
    rf_model = pickle.load(file)
with open("lr_model_pca_tuned.pkl", "rb") as file:
    lr_model = pickle.load(file)
with open("knn_model_pca_tuned.pkl", "rb") as file:
    knn_model = pickle.load(file)

# Load the scaler and PCA
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
with open("pca.pkl", "rb") as file:
    pca = pickle.load(file)


def preprocess_image(image):
    """
    Preprocess the image: resize, flatten, scale, and apply PCA.
    """
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))  # Resize to match the training input
    image_array = np.array(image).flatten()

    # Scale the image
    image_scaled = scaler.transform([image_array])

    # Apply PCA
    image_pca = pca.transform(image_scaled)

    return image_pca[0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/index.html")
def index2():
    return render_template("index.html")


@app.route("/patientstories.html")
def patientStories():
    return render_template("patientstories.html")


@app.route("/know_symptoms.html")
def know_symptoms():
    return render_template("know_symptoms.html")


@app.route("/datasetdetails.html")
def datasetDetails():
    return render_template("datasetdetails.html")


@app.route("/terms2.html")
def terms():
    return render_template("terms2.html")


@app.route("/predict.html")
def predict2():
    return render_template("predict.html")


@app.route("/contactUs.html")
def contactus():
    return render_template("contactUs.html")


@app.route("/sf.html")
def sf():
    return render_template("sf.html")


@app.route("/su.html")
def su():
    return render_template("su.html")


@app.route("/researchDetail.html")
def researchDetails():
    return render_template("researchDetail.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Ensure that a file is received
    if "file" not in request.files:
        return render_template("error.html", error="No file part")

    file = request.files["file"]
    if file.filename == "":
        return render_template("error.html", error="No selected file")

    selected_model = request.form["model-select"]
    print(selected_model)

    try:
        image = file.read()
        processed_image = preprocess_image(image)

        if selected_model == "model1":
            probabilities = xgb_model.predict_proba([processed_image])
            probability_of_cancer = probabilities[0][1]

        if selected_model == "model4":
            probabilities = lr_model.predict_proba([processed_image])
            probability_of_cancer = probabilities[0][1]

        if selected_model == "model2":
            probabilities = rf_model.predict_proba([processed_image])
            probability_of_cancer = probabilities[0][1]

        if selected_model == "model3":
            probabilities = knn_model.predict_proba([processed_image])
            probability_of_cancer = probabilities[0][1]

        if probability_of_cancer > 0.5:
            message = (
                "Increased Risk Detected - Please Consult a Healthcare Professional."
            )
            color = "red"
        else:
            message = "Low Risk Detected - No Immediate Concerns."
            color = "green"

        return render_template(
            "predict.html",
            probability=probability_of_cancer,
            message=message,
            color=color,
        )
    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)

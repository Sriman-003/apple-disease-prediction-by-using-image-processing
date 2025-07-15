import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = "best_model.h5"
model = load_model(MODEL_PATH)

# Image processing parameters
IMAGE_SIZE = (224, 224)

# Class labels, descriptions, and pesticide suggestions
class_labels = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]
disease_descriptions = {
    "Blotch_Apple": "Blotch appears as dark, irregular spots on the apple's surface, caused by fungal infections.",
    "Normal_Apple": "This apple is healthy with no signs of disease.",
    "Rot_Apple": "Rot is a sign of decay, often caused by bacterial or fungal infections.",
    "Scab_Apple": "Scab results in rough, dark patches on the apple, affecting its quality and shelf life."
}
pesticide_suggestions = {
    "Blotch_Apple": "Use fungicides like Captan or Mancozeb to control fungal blotch.",
    "Normal_Apple": "No pesticide needed. Maintain good orchard hygiene to prevent diseases.",
    "Rot_Apple": "Apply fungicides such as Thiophanate-methyl or Iprodione to manage rot.",
    "Scab_Apple": "Use fungicides like Sulfur or Myclobutanil to treat apple scab."
}

def predict_image(image_path):
    """
    Predict the disease from an uploaded image.
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Make prediction
    prediction = model.predict(x)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100  # Convert to percentage
    predicted_class = class_labels[predicted_index]
    description = disease_descriptions.get(predicted_class, "No description available.")
    pesticide = pesticide_suggestions.get(predicted_class, "No pesticide recommendation available.")

    return predicted_class, confidence, description, pesticide


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Home page: Handles file upload and redirects to the result page.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Get prediction
        predicted_class, confidence, description, pesticide = predict_image(filepath)

        # Redirect to the result page with prediction data
        return redirect(url_for("result", filename=file.filename, prediction=predicted_class, confidence=confidence, description=description, pesticide=pesticide))

    return render_template("index.html")


@app.route("/result")
def result():
    """
    Result page: Displays the prediction results.
    """
    # Retrieve prediction data from query parameters
    filename = request.args.get("filename")
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    description = request.args.get("description")
    pesticide = request.args.get("pesticide")

    return render_template("result.html", filename=filename, prediction=prediction, confidence=confidence, description=description, pesticide=pesticide)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
# import pickle
# import numpy as np
# from PIL import Image
# import os
# from sklearn.preprocessing import StandardScaler


# def preprocess_image(image_path):
#     """
#     Preprocess the image to match the model's input requirements.
#     Replace this with your actual preprocessing steps.
#     """
#     with Image.open(image_path) as image:
#         # img = img.resize((224, 224))  # Example resize, adjust as needed
#         image = image.resize((224, 224))  # Resize to match the training input
#         s = StandardScaler()
#         image_array = s.fit_transform(image)
#         img_arr = np.array(image_array).flatten()
#         # img_arr = img_arr / 255.0  # Example normalization, adjust as needed
#     return img_arr


# def load_model(model_path):
#     """
#     Load the trained model from the specified path.
#     """
#     with open(model_path, "rb") as file:
#         model = pickle.load(file)
#     return model


# def main():
#     # Path to your model and test images
#     model_path = "xgb_model_pca_tuned.pkl"
#     test_images_dir = "breastcancer/train/resized_noisy/"

#     # Load the model
#     model = load_model(model_path)

#     # Threshold for classifying a sample as 'sick'
#     threshold = 0.3

#     # Load and preprocess test images
#     test_images = os.listdir(test_images_dir)
#     for image_name in test_images:
#         image_path = os.path.join(test_images_dir, image_name)
#         processed_image = preprocess_image(image_path)

#         # Get probability estimates
#         proba = model.predict_proba([processed_image])

#         # Apply the threshold to get the predicted class
#         # Assuming that the second column ([1]) is the probability of being 'sick'
#         prediction = (proba[:, 1] >= threshold).astype(int)

#         print(
#             f"Image: {image_name}, Prediction: {'Sick' if prediction[0] == 1 else 'Healthy'}"
#         )


# if __name__ == "__main__":
#     main()


# import pickle
# import numpy as np
# from PIL import Image
# import os
# import csv


# def preprocess_image(image_path):
#     """
#     Preprocess the image to match the model's input requirements.
#     Replace this with your actual preprocessing steps.
#     """
#     with Image.open(image_path) as img:
#         img = img.resize((224, 224))  # Example resize, adjust as needed
#         img_arr = np.array(img).flatten()
#         img_arr = img_arr / 255.0  # Example normalization, adjust as needed
#     return img_arr


# def load_model(model_path):
#     """
#     Load the trained model from the specified path.
#     """
#     with open(model_path, "rb") as file:
#         model = pickle.load(file)
#     return model


# def main():
#     # Path to your model and test images
#     model_path = "xgb_model.pkl"
#     test_images_dir = "breastcancer/train/resized_noisy/"
#     output_csv = "predictions.csv"  # Output CSV file

#     # Load the model
#     model = load_model(model_path)

#     # Threshold for classifying a sample as 'sick'
#     threshold = 0.5

#     # Open CSV file for writing
#     with open(output_csv, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         # Write header
#         writer.writerow(["Image", "Prediction"])

#         # Load and preprocess test images
#         test_images = os.listdir(test_images_dir)
#         for image_name in test_images:
#             image_path = os.path.join(test_images_dir, image_name)
#             processed_image = preprocess_image(image_path)

#             # Get probability estimates
#             proba = model.predict_proba([processed_image])

#             # Apply the threshold to get the predicted class
#             prediction = (proba[:, 1] >= threshold).astype(int)
#             prediction_label = "Sick" if prediction[0] == 1 else "Healthy"

#             # Write the prediction to the CSV file
#             writer.writerow([image_name, prediction_label])

#             # Print the prediction
#             print(f"Image: {image_name}, Prediction: {prediction_label}")


# if __name__ == "__main__":
#     main()


import pickle
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess_image(image_path):
    """
    Preprocess the image to only resize and flatten.
    """
    with Image.open(image_path) as image:
        image = image.resize((224, 224))  # Resize to match the training input
        img_arr = np.array(image).flatten()
    return img_arr


def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_scaler_pca(scaler_path, pca_path):
    """
    Load the saved scaler and PCA model.
    """
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
    with open(pca_path, "rb") as file:
        pca = pickle.load(file)
    return scaler, pca


def main():
    # Paths
    model_path = "xgb_model_pca_tuned.pkl"
    scaler_path = "scaler.pkl"  # Path to your saved scaler
    pca_path = "pca.pkl"  # Path to your saved PCA model
    test_images_dir = "breastcancer/train/resized_noisy/"

    # Load the model, scaler, and PCA
    model = load_model(model_path)
    scaler, pca = load_scaler_pca(scaler_path, pca_path)

    # Threshold for classifying a sample as 'sick'
    threshold = 0.3

    # Process and predict for each test image
    test_images = os.listdir(test_images_dir)
    for image_name in test_images:
        image_path = os.path.join(test_images_dir, image_name)
        processed_image = preprocess_image(image_path)

        # Scale and apply PCA
        processed_image_scaled = scaler.transform([processed_image])
        processed_image_pca = pca.transform(processed_image_scaled)

        # Predict
        proba = model.predict_proba(processed_image_pca)
        prediction = (proba[:, 1] >= threshold).astype(int)
        print(
            f"Image: {image_name}, Prediction: {'Sick' if prediction[0] == 1 else 'Healthy'}"
        )


if __name__ == "__main__":
    main()

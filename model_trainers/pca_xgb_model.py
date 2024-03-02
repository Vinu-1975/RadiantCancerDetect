import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
import pickle
import os
import csv
from PIL import Image

warnings.filterwarnings("ignore")


base_dir = "breastcancer"
base_dir = os.path.join(base_dir,"breastcancer")
train_dir = os.path.join(base_dir, "train")
resized_dir = os.path.join(train_dir, "resized_noisy")
csv_filename = "images_labels_resized.csv"

flattened_images_list = []
labels_list = []

with open(csv_filename, "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header

    for row in csv_reader:
        image_name, label = row
        image_path = os.path.join(resized_dir, image_name)

        with Image.open(image_path) as img:
            img_arr = np.array(img)
            flattened_img_arr = img_arr.flatten()  # Flatten the image
            flattened_images_list.append(flattened_img_arr)
            labels_list.append(int(label))

x = np.array(flattened_images_list)
y = np.array(labels_list)

# Scale the features
s = StandardScaler()
x_scaled = s.fit_transform(x)
with open("scaler.pkl", "wb") as file:
    pickle.dump(s, file)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.33, shuffle=True, random_state=42
)

# Apply PCA
pca = PCA(n_components=178)
x_train_pca = pca.fit_transform(x_train)
with open("pca.pkl", "wb") as file:
    pickle.dump(pca, file)
x_test_pca = pca.transform(x_test)

# Create and train the XGBoost model with specified hyperparameters
model = xgb.XGBClassifier(
    objective="binary:logistic",
    colsample_bytree=0.6733618039413735,
    gamma=0.1216968971838151,
    learning_rate=0.16217936517334897,
    max_depth=int(5.159725093210579),
    n_estimators=int(93.68437102970628),
)
model.fit(x_train_pca, y_train)

# Evaluate the model
accuracy = model.score(x_test_pca, y_test)
print(f"Model Accuracy: {accuracy}")

# Save the model
with open("xgb_model_pca_tuned.pkl", "wb") as file:
    pickle.dump(model, file)

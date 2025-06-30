import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
import random
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
tic=time.time()

# === CONFIG ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 40
DATA_DIR = 'Images dataset4'
VALIDATION_SPLIT = 0.1


# === MODEL BUILDER ===
def build_model(num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='swish', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='swish'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='swish'),
        MaxPooling2D(2, 2),
        Conv2D(256, (5, 5), activation='swish'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(248, activation='relu'),
        Dense(64, activation="relu"),
        Dropout(0.08),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === DATA PREP ===
all_paths = glob(f"{DATA_DIR}/*/*/*.*")
random.shuffle(all_paths)
val_count = int(len(all_paths) * VALIDATION_SPLIT)
val_paths = all_paths[:val_count]
train_paths = all_paths[val_count:]

def extract_labels(path):
    parts = path.split(os.sep)
    plant = parts[-3]
    disease = parts[-2]
    return plant, disease

plants = sorted(set(extract_labels(p)[0] for p in all_paths))
plant_to_idx = {name: i for i, name in enumerate(plants)}

disease_sets = defaultdict(set)
for path in all_paths:
    plant, disease = extract_labels(path)
    disease_sets[plant].add(disease)

# === STAGE 1: PLANT CLASSIFICATION ===
print("\n Training Plant Classifier")
train_imgs = []
train_plant_labels = []

for path in train_paths:
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    plant, _ = extract_labels(path)
    train_imgs.append(img)
    train_plant_labels.append(plant_to_idx[plant])

train_imgs = np.array(train_imgs)
train_plant_labels = to_categorical(train_plant_labels, num_classes=len(plants))

plant_model = build_model(num_classes=len(plants))
history_plant = plant_model.fit(
    train_imgs, train_plant_labels,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=13)
    ]
)

# === STAGE 2: DISEASE MODELS ===
disease_models = {}
disease_class_maps = {}

print("\n Training Disease Classifiers for Each Plant")
for plant in plants:
    plant_train_paths = [p for p in train_paths if extract_labels(p)[0] == plant]
    disease_list = sorted(disease_sets[plant])
    dis_to_idx = {dis: i for i, dis in enumerate(disease_list)}

    X, y = [], []
    for path in plant_train_paths:
        img = load_img(path, target_size=IMG_SIZE)
        img = img_to_array(img) / 255.0
        _, disease = extract_labels(path)
        X.append(img)
        y.append(dis_to_idx[disease])
    
    X = np.array(X)
    y = to_categorical(y, num_classes=len(disease_list))

    model = build_model(len(disease_list))
    model.fit(
        X, y,
        validation_split=0.1,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=13)
        ]
    )

    disease_models[plant] = model
    disease_class_maps[plant] = {v: k for k, v in dis_to_idx.items()}


# === VALIDATION + STAGE 2 METRICS ===
print("\n Validating Full Hierarchical Model")
true_labels = []
pred_labels = []
true_diseases = []
pred_diseases = []
correct = 0

for path in val_paths:
    plant_gt, disease_gt = extract_labels(path)

    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_plant_idx = np.argmax(plant_model.predict(img_array, verbose=0))
    pred_plant = plants[pred_plant_idx]

    disease_model = disease_models[pred_plant]
    disease_map = disease_class_maps[pred_plant]
    pred_disease_idx = np.argmax(disease_model.predict(img_array, verbose=0))
    pred_disease = disease_map[pred_disease_idx]

    true = f"{plant_gt}-{disease_gt}"
    pred = f"{pred_plant}-{pred_disease}"

    true_labels.append(true)
    pred_labels.append(pred)

    true_diseases.append(true)
    pred_diseases.append(pred)

    if true == pred:
        correct += 1


accuracy = correct / len(val_paths)
print(f" Final Accuracy: {correct}/{len(val_paths)} = {accuracy:.2%}")


# === STAGE 2: REPORTS AND VISUALIZATION ===
print(" Stage 2: Disease Classification Report")
print(classification_report(true_diseases, pred_diseases))

labels = sorted(set(true_diseases + pred_diseases))
cm = confusion_matrix(true_diseases, pred_diseases, labels=labels)

toc=time.time()

# Confusion Matrix Plot
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title("Confusion Matrix: True vs Predicted Diseases (All Plants)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Per-Class Accuracy Plot
per_class_accuracy = (cm.diagonal() / cm.sum(axis=1)) * 100
plt.figure(figsize=(12, 6))
plt.barh(labels, per_class_accuracy)
plt.title("Per-Class Accuracy (Disease Level)")
plt.xlabel("Accuracy (%)")
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

# Stage 1 Accuracy Plot
plt.figure(figsize=(8, 4))
plt.plot(history_plant.history['accuracy'], label='Train Accuracy')
plt.plot(history_plant.history['val_accuracy'], label='Val Accuracy')
plt.title("Plant Classifier Accuracy (Stage 1)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("time taken by heap form3:",tic-toc)




import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Error enabling memory growth:", e)

import os
import numpy as np
from tensorflow.keras import layers
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import time
import gc
from imblearn.over_sampling import SMOTE
import h5py
import argparse

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="CNN for Land Cover Classification with Center-Pixel Prediction")
parser.add_argument('--input_h5', type=str, default='D:/DCEC/30m_data_tiles_2023.h5',
                    help='Path to input .h5 file containing data, i, j, label')
parser.add_argument('--blueprint', type=str, default='D:/DCEC/MKInput_30m_f16.tif',
                    help='Path to blueprint raster for georeferencing')
parser.add_argument('--subset_size', type=int, default=6024, choices=[500, 1000, 2000, 4000, 6024],
                    help='Number of labeled samples to use (500, 1000, 2000, 4000, 6024)')
parser.add_argument('--batch_size', type=int, default=50000, help='Batch size for prediction')
parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='D:/Mekong_supervised', help='Output directory for results and map')

args = parser.parse_args()

# -----------------------------
# Settings from args
# -----------------------------
input_h5 = args.input_h5
blueprint_path = args.blueprint
output_dir = args.output_dir

patch_size = 48
bands = 10
num_classes = 12

chosen_subset = args.subset_size
train_batch_size = args.train_batch_size
pred_batch_size = args.batch_size
epochs = args.epochs
lr = args.lr

start_time = time.time()

# -----------------------------
# Load all data from .h5
# -----------------------------
print(f"Loading data from {input_h5}...")
with h5py.File(input_h5, 'r') as f:
    labels = f['label'][:]                         # mode label
    mask = labels != 0
    labeled_indices = np.where(mask)[0]
    if len(labeled_indices) < chosen_subset:
        raise ValueError(f"Only {len(labeled_indices)} labeled samples found, need {chosen_subset}")
    subset_idx = np.random.choice(labeled_indices, chosen_subset, replace=False)
    subset_idx = np.sort(subset_idx)
    data_labeled = f['data'][subset_idx].astype(np.float32)  # (subset, 48, 48, 10)
    labels_labeled = labels[subset_idx]
    print(f"Loaded {data_labeled.shape[0]:,} labeled tiles")
# -----------------------------
# Normalize the ENTIRE dataset first (your original logic)
# -----------------------------
# -----------------------------
# Train/validation split (70/30)
# -----------------------------
prop_train = 0.7
n_total = len(data_labeled)
n_train = int(n_total * prop_train)

perm = np.random.permutation(n_total)
x_train = data_labeled[perm[:n_train]]
y_train_raw = labels_labeled[perm[:n_train]]
x_val = data_labeled[perm[n_train:]]
y_val_raw = labels_labeled[perm[n_train:]]

# -----------------------------
# SMOTE on training set
# -----------------------------
print("Applying SMOTE...")
x_train_flat = x_train.reshape(x_train.shape[0], -1)
smote = SMOTE(random_state=42, k_neighbors=5)
x_train_resampled_flat, y_train_resampled = smote.fit_resample(x_train_flat, y_train_raw - 1)
x_train = x_train_resampled_flat.reshape(-1, patch_size, patch_size, bands)

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train_resampled, num_classes)
y_val = tf.keras.utils.to_categorical(y_val_raw - 1, num_classes)

print(f"After SMOTE → Train: {x_train.shape[0]}, Val: {x_val.shape[0]}")

# -----------------------------
# Model definition
# -----------------------------
def cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(64, (3, 3), activation=None, padding='same')(inputs)
    bn1 = layers.BatchNormalization()(conv1)
    relu1 = layers.ReLU()(bn1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(relu1)

    conv2 = layers.Conv2D(128, (3, 3), activation=None, padding='same')(pool1)
    bn2 = layers.BatchNormalization()(conv2)
    relu2 = layers.ReLU()(bn2)

    gap = layers.GlobalAveragePooling2D()(relu2)
    dense1 = layers.Dense(128, activation='relu')(gap)
    dropout1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(128, activation='relu')(dropout1)
    outputs = layers.Dense(num_classes, activation='softmax')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = cnn_model((patch_size, patch_size, bands), num_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# -----------------------------
# Training
# -----------------------------
history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=train_batch_size,
    verbose=1
)

# -----------------------------
# Validation metrics & plots
# -----------------------------
# Loss plot
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(output_dir, f'train_val_loss_{chosen_subset}.png'))
plt.close()

# Predictions on validation
y_pred_val = model.predict(x_val)
pred_classes = np.argmax(y_pred_val, axis=1)
true_classes = y_val_raw - 1

f1 = f1_score(true_classes, pred_classes, average='macro')
accuracy = accuracy_score(true_classes, pred_classes)
precision = precision_score(true_classes, pred_classes, average='macro')
recall = recall_score(true_classes, pred_classes, average='macro')
kappa = cohen_kappa_score(true_classes, pred_classes)

print(f"F1: {f1:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, Kappa: {kappa:.4f}")

# Confusion matrix
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=pred_classes, yticklabels=pred_classes, cbar=True)
plt.title('CNN Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(output_dir, f'confusion_matrix_{chosen_subset}.png'))
plt.close()

# -----------------------------
# Center-pixel prediction using already-loaded data
# -----------------------------
def predict_whole_area_chunked(model, h5_path, blueprint_path, output_path, batch_size):
    with rasterio.open(blueprint_path) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()

    predicted_image = np.full((height, width), -1, dtype=np.int8)

    with h5py.File(h5_path, 'r') as f:
        n_tiles = f['data'].shape[0]
        i_positions = f['i'][:]
        j_positions = f['j'][:]

        print(f"Predicting on {n_tiles:,} tiles in batches of {batch_size}...")

        for start in range(0, n_tiles, batch_size):
            end = min(start + batch_size, n_tiles)
            batch_data = f['data'][start:end].astype(np.float32)  # (batch, 48, 48, 10)

            # Predict
            preds = model.predict(batch_data, verbose=0)
            classes = np.argmax(preds, axis=1)

            # Map to center pixels
            center_r = i_positions[start:end]
            center_c = j_positions[start:end]
            valid = (center_r < height) & (center_c < width)
            predicted_image[center_r[valid], center_c[valid]] = (classes[valid] + 1).astype(np.int8)

            # Cleanup
            del batch_data, preds, classes
            gc.collect()

            if (start // batch_size + 1) % 50 == 0:
                print(f"  Processed {end:,}/{n_tiles:,} tiles")

    # Write output raster
    profile.update(dtype=rasterio.int8, count=1, nodata=-1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predicted_image, 1)

    print(f"Prediction map saved to: {output_path}")

# Output filename with subset size
output_raster_path = os.path.join(output_dir, f'CNN_predicted_output_{chosen_subset}.tif')

# Run prediction
predict_whole_area_chunked(model, input_h5, blueprint_path, output_raster_path, pred_batch_size)

# -----------------------------
# Save results
# -----------------------------
total_time = time.time() - start_time
results = {
    'Subset size': chosen_subset,
    'Accuracy': accuracy,
    'Kappa': kappa,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Training time (s)': total_time
}

results_file = os.path.join(output_dir, f'model_results_{chosen_subset}.txt')
with open(results_file, 'w') as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")

print(f"Results saved to {results_file}")
print(f"Total runtime: {total_time:.2f} seconds")

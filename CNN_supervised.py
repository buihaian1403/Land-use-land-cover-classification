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
import pandas as pd
from rasterio.transform import from_origin
import time
import gc
# Specify the path to the main folder containing image tiles and labels
inputNPZ = 'D:/DCEC/HighRes_data_tiles_2023.npz'
label_csv = 'D:/DCEC/HighRes_data_tiles_2023_ij_included.csv'

prop_train, prop_val = 0.8, 0.2  # Proportions for train/validation split
patch_size = 24
bands = 10  # Adjust to match the data
num_classes = 12  # Number of output classes
start_time = time.time()
# Load image data
with np.load(inputNPZ, allow_pickle=False) as npz_file:
    tiles = dict(npz_file.items())
    data = tiles['data'].astype(np.float32)  # Convert to float32 for processing
    input_dim = data.shape[1:4]  # Shape of the input images
    del tiles
    gc.collect()

# Load labels
n_bands = data.shape[3]
for band in np.arange(n_bands):
    band_data = data[:, :, :, band]
    invalid_mask = (band_data == -32768) | np.isnan(band_data)
    valid_values = band_data[~invalid_mask]
    mean_val = np.mean(valid_values)
    band_data[invalid_mask] = mean_val
    max_val = np.max(band_data)
    min_val = np.min(band_data)
    data[:,:,:,band] = (band_data - min_val)/(max_val - min_val)

labels = np.genfromtxt(label_csv, delimiter=',', skip_header=1, usecols=2, dtype=int)

# Mask and extract labeled images
mask = labels != 0  # Identify labeled images (label != 0)
data_labeled = data[mask]  # Extract only labeled images
labels_labeled = labels[mask]  # Extract corresponding labels

# Standardize pixel values between 0 and 1

# Split labeled data into train, validation, and test datasets
# Proportions for train/validation split
total_labeled = data_labeled.shape[0]

# Compute train/validation sizes
size_train = int(total_labeled * prop_train)
size_val = total_labeled - size_train  # Complement of training size

# Shuffle data
permuted_indices = np.random.permutation(total_labeled)
data_labeled = data_labeled[permuted_indices]
labels_labeled = labels_labeled[permuted_indices]

# Split data (non-overlapping)
x_train, x_val = data_labeled[:size_train], data_labeled[size_train:]
y_train, y_val = labels_labeled[:size_train], labels_labeled[size_train:]

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes)  # Adjust label range for one-hot encoding
y_val = tf.keras.utils.to_categorical(y_val - 1, num_classes)

# Print the shapes of the training and validating sets for verification
print('X Train shape: ', x_train.shape)
print('Y Train shape: ', y_train.shape)
print('X validating shape: ', x_val.shape)
print('Y validating shape: ', y_val.shape)

# Define the CNN model
def cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer with batch normalization
    conv1 = layers.Conv2D(64, (3, 3), activation=None, padding='same')(inputs)
    bn1 = layers.BatchNormalization()(conv1)
    relu1 = layers.ReLU()(bn1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(relu1)

    # Second convolutional layer
    conv2 = layers.Conv2D(128, (3, 3), activation=None, padding='same')(pool1)
    bn2 = layers.BatchNormalization()(conv2)
    relu2 = layers.ReLU()(bn2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(relu2)

    # Third convolutional layer
    conv3 = layers.Conv2D(256, (3, 3), activation=None, padding='same')(pool2)
    bn3 = layers.BatchNormalization()(conv3)
    relu3 = layers.ReLU()(bn3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(relu3)

    # Global Average Pooling (GAP)
    gap = layers.GlobalAveragePooling2D()(pool3)

    # Dense layers
    dense1 = layers.Dense(128, activation='relu')(gap)
    dropout1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(128, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    dense3 = layers.Dense(64, activation='relu')(dropout2)
    outputs = layers.Dense(num_classes, activation='softmax')(dense3)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define the input shape based on the shape of the training data (excluding the batch size)
input_shape = x_train.shape[1:]

# Create an instance of the CNN model using the defined input shape and specifying the number of classes
model = cnn_model(input_shape, num_classes=num_classes)

# Display a summary of the model architecture
model.summary()

# Compile the model with specified configuration for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model on the training data
history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=100
)

# Plot training and validating loss
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(np.array(history.history['loss']), label='Training Loss')
plt.plot(np.array(history.history['val_loss']), label='Validating Loss')
plt.ylabel('Loss Function')
plt.xlabel('Epochs')
plt.legend(['Train', 'validating'], loc='best')
plt.savefig('train_val loss.png')
plt.close()

# Make predictions using the trained model on the validating data
y_pred = model.predict(x_val)

# Display the shape of the predicted values
print("Shape of Predicted Values:", y_pred.shape)
print(y_pred)
prediction = np.argmax(y_pred, axis=1)
print(prediction)
actual = np.argmax(y_val, axis=1)

# Calculate metrics
f1 = f1_score(actual, prediction, average='macro')
accuracy = accuracy_score(actual, prediction)
precision = precision_score(actual, prediction, average='macro')
recall = recall_score(actual, prediction, average='macro')
kappa = cohen_kappa_score(actual, prediction)

# Print metrics
print('F1 Score:', f1)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('Kappa:', kappa)

        
# Confusion matrix
cmat = confusion_matrix(actual, prediction)

# Plot confusion matrix
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(cmat, annot=True, fmt='d', cmap='RdYlGn',  yticklabels=num_classes, xticklabels=num_classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_train_val.png')
plt.close()
gc.collect()
def predict_whole_area(model, data, csv_path, blueprint_raster_path, output_path, patch_size):
    # Load positions (i,j) from the CSV file
    csv_data = pd.read_csv(csv_path)
    i_positions = csv_data.iloc[:, 0].astype(int)  # Row positions for each image
    j_positions = csv_data.iloc[:, 1].astype(int)  # Column positions for each image

    # Load blueprint raster metadata
    with rasterio.open(blueprint_path) as src:
        blueprint_transform = src.transform
        blueprint_crs = src.crs
        blueprint_width = src.width
        blueprint_height = src.height
        Xmin, Ymax = src.bounds.left, src.bounds.top
        blueprint_res_x, blueprint_res_y = src.res
    new_res_x = blueprint_res_x * patch_size
    new_res_y = blueprint_res_y * patch_size
    new_width = blueprint_width // patch_size
    new_height = blueprint_height // patch_size
    new_transform = from_origin(Xmin, Ymax, new_res_x, new_res_y)
    
# Update the transformation matrix for the new resolution
    predicted_image = np.full((new_height, new_width), -1,dtype=np.int8)
    # Process each image and map its prediction onto the predicted image
    batch_size = 2400  # Adjust this batch size based on GPU memory constraints
    for batch_idx in range(0, len(data), batch_size):
    # Iterate through batches for GPU inference
        batch_patches = data[batch_idx:batch_idx + batch_size]
        predictions = model.predict(batch_patches)
        print(predictions)
        # Get the predicted class labels
        predicted_classes = np.argmax(predictions, axis=1)
        
        for idx, predicted_class in enumerate(predicted_classes):
            i = i_positions[batch_idx + idx]
            j = j_positions[batch_idx + idx]
            
            predicted_image[i//patch_size, j//patch_size] = predicted_class  # Adjust label to 1-based
        
        gc.collect() 
    # Update metadata for the output raster
    with rasterio.open(blueprint_raster_path) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": new_height,
            "width": new_width,
            "count": 1,
            "dtype": 'int8',
            "crs": blueprint_crs,
            "transform": new_transform,
            "nodata": -1
        })
    # Save the predicted image as a new raster file
    with rasterio.open(output_path, 'w', **out_meta) as out_ds:
        out_ds.write(predicted_image, 1)

    print(f"Prediction raster saved to {output_path}")

# Paths to input and output raster files
blueprint_path = 'D:\DCEC\MKInput_5m.tif'
output_raster_path = 'CNN_predicted_output.tif'

# Perform prediction
predict_whole_area(model, data, label_csv, blueprint_path, output_raster_path, patch_size)

end_time = time.time()
total_time = end_time - start_time
print(total_time)

Results = {'Accuracy:': accuracy,
           'Kappa:': kappa,
           'Precision:': precision,
           'Recall:': recall, 
           'F1 Score:': f1,
           'time consummed:': total_time}

with open('model_results.txt', 'w') as f:
    for metric, value in Results.items():
        f.write(f"{metric} {value}\n")
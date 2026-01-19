import cupy as cp
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import json
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import time
import gc
# Load input data from raster file
input_path = 'MKInput_30m_f16.tif'
nodata = -32768
subset_size = 6024
start_time = time.time()

with rasterio.open(input_path) as ds:
    labels = ds.read(1).astype(np.int16)
    img = ds.read(list(range(2,12))).astype(np.int16)
    img = np.transpose(img, (1, 2, 0))
    meta = ds.meta.copy()
    nodata = ds.nodata if ds.nodata is not None else nodata

img = img.astype(np.float32)  # Convert to float for proper NaN handling
img[img == nodata] = np.nan
scaler = StandardScaler()
# Fill nodata values with band-wise mean
for band in range(img.shape[2]):
    band_data = img[:, :, band]
    mask = (band_data != nodata) & (~np.isnan(band_data))
    band_mean = band_data[mask].mean()
    band_data[~mask] = band_mean
    band_data = scaler.fit_transform(band_data.reshape(-1, 1)).flatten()
    img[:, :, band] = band_data.reshape(img.shape[0], img.shape[1])

img = img.astype(np.float32)
print(f'We are working with {img.shape[2]} features')

# ==================== EXTRACT LABELED PIXELS AND SPLIT ====================
labeled_mask = (labels > 0) & (labels < 13)
n_labeled = labeled_mask.sum()
print(f'Total labeled pixels: {n_labeled}')

classes = np.unique(labels[labeled_mask])
print(f'Classes ({len(classes)}): {classes}')

# Extract features and labels from labeled pixels
X_all = img[labeled_mask]
y_all = labels[labeled_mask]

selected_indices = np.random.choice(len(y_all), size=subset_size, replace=False)
X_subset = X_all[selected_indices]
y_subset = y_all[selected_indices]
# Stratified split: 70% train+val, 30% test
X, X_test, y, y_test = train_test_split(X_subset, y_subset, test_size=0.30, random_state=42, shuffle=True )

print(f'Training + validation pixels: {len(y)}')
print(f'Test pixels: {len(y_test)}')

# Handle class imbalance using ADASYN and RandomUndersample
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X.reshape(X.shape[0], -1), y)

# Convert to CuPy arrays
X_cp = cp.asarray(X_resampled, dtype=cp.float32)
y_cp = cp.asarray(y_resampled, dtype=cp.int32)
X_test_cp = cp.asarray(X_test.reshape(X_test.shape[0], -1), dtype=cp.float32)
y_test_cp = cp.asarray(y_test, dtype=cp.int32)

# Define objective function for Hyperopt with cross-validation
def rf_objective(params):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    
    # Add checks to ensure the parameters are valid
    if n_estimators <= 0 or max_depth <= 0:
        return {'loss': float('inf'), 'status': STATUS_OK}
    rf_model = cuRF(n_estimators=n_estimators, max_depth=max_depth)
    scores = cross_val_score(rf_model, X_np, y_np, cv=5, scoring='accuracy')
    accuracy = scores.mean()
    return {'loss': -accuracy, 'status': STATUS_OK}

def svm_objective(params):
    C = params['C']
    gamma = params['gamma']
    
    # Add checks to ensure the parameters are valid
    if C <= 0 or gamma <= 0:
        return {'loss': float('inf'), 'status': STATUS_OK}
    
    svm_model = SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(svm_model, X_np, y_np, cv=5, scoring='accuracy')
    accuracy = scores.mean()
    return {'loss': -accuracy, 'status': STATUS_OK}

# Hyperopt parameter spaces
rf_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 20),  # Uniform distribution with a step size of 10
    'max_depth': hp.quniform('max_depth', 5,20, 1)  # Uniform distribution from 5 to 20 with step size of 1
}

svm_space = {
    'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),  # Log uniform distribution for C
    'gamma': hp.loguniform('gamma', np.log(1e-3), np.log(1e3))  # Log uniform distribution for gamma
}

# Evaluate models
def evaluate_final_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    return accuracy, kappa, f1, precision, recall, conf_matrix

# Predict for each pixel on full image using best models
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array_np = img.reshape(new_shape)
img_as_array = cp.asarray(img_as_array_np)

def predict_in_chunks(model, data_cp, chunk_size=100000):
    n = data_cp.shape[0]
    pred = []
    for i in range(0, n, chunk_size):
        chunk = data_cp[i:i+chunk_size]
        pred_chunk = model.predict(chunk)
        pred.append(pred_chunk)
        del chunk, pred_chunk
        gc.collect()
    return np.concatenate(pred)

def save_classification_map(filename, data, meta):
    data = data.astype(np.int8)
    meta.update({
        'dtype': 'uint8',
        'count': 1,
        'height': data.shape[0],
        'width': data.shape[1],
        'transform': meta['transform']  # Use meta's transform
    })
    if 'nodata' in meta:
        meta['nodata'] = None
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data, 1)

# Plot and save confusion matrices
def plot_confusion_matrix(cm, title, filename, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

# Convert to cuPy
X_np = cp.asnumpy(X_cp)
y_np = cp.asnumpy(y_cp)
X_test_np = cp.asnumpy(X_test_cp)
y_test_np = cp.asnumpy(y_test_cp)

# Optimize Random Forest
rf_trials = Trials()
best_rf = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=100, trials=rf_trials)
best_rf_params = rf_trials.best_trial['result']

# Train final models with best parameters
rf_model = cuRF(n_estimators=int(best_rf['n_estimators']),
                max_depth=int(best_rf['max_depth']))
rf_model.fit(X_np, y_np)
rf_accuracy_test, rf_kappa_test, rf_f1, rf_precision, rf_recall, rf_conf_matrix = evaluate_final_model(rf_model, X_test_np, y_test_np)
result = permutation_importance(rf_model, X_np, y_np, n_repeats=10, random_state=42)
importance = result.importances_mean
for i, v in enumerate(importance):
    print('Feature: %d, Score: %.5f' % (i, v))
plt.bar([x for x in range(len(importance))], importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.savefig('feature_importances.png')
plt.close()  
print(f"Random Forest Accuracy: {rf_accuracy_test}")
print(f"Random Forest Kappa: {rf_kappa_test}")
print(f"Random Forest f1: {rf_f1}")
print(f"Random Forest recall: {rf_recall}")
print(f"Random Forest precision: {rf_precision}")
print("RF Confusion Matrix:\n", rf_conf_matrix)

class_prediction_rf = rf_model.predict(img_as_array).get()
class_prediction_rf = class_prediction_rf.reshape(img[:, :, 0].shape)
save_classification_map(f'RF_classification_map_{subset_size}.tif', class_prediction_rf, meta)
plot_confusion_matrix(rf_conf_matrix, f'Random Forest Confusion Matrix', f'RF_confusion_matrix_{subset_size}.png', classes)
end_time_RF = time.time() 
RF_time = end_time_RF - start_time
print(f'RF prediction map saved as RF_classification_map.tif')
print(f'Time consume: {RF_time} seconds')
del rf_model, X_resampled, y_resampled, X_cp, y_cp, X_test_cp, y_test_cp
gc.collect
cp.get_default_memory_pool().free_all_blocks()

start_time2 = time.time()

# Optimize SVM
svm_trials = Trials()
best_svm = fmin(fn=lambda params: svm_objective(params), space=svm_space, algo=tpe.suggest, max_evals=100, trials=svm_trials)
best_svm_params = best_svm

#Training SVM model
svm_model = SVC(C=best_svm_params['C'], gamma=best_svm_params['gamma'], kernel='rbf')
svm_model.fit(X_np, y_np)
svm_accuracy_test, svm_kappa_test, svm_f1, svm_precision, svm_recall, svm_conf_matrix = evaluate_final_model(svm_model, X_test_np, y_test_np)

print(f"SVM Accuracy: {svm_accuracy_test}")
print(f"SVM Kappa: {svm_kappa_test}")
print(f"SVM f1: {svm_f1}")
print(f"SVM recall: {svm_recall}")
print(f"SVM precision: {svm_precision}")
print("SVM Confusion Matrix:\n", svm_conf_matrix)

class_prediction_svm = predict_in_chunks(svm_model, img_as_array_np, chunk_size=100000)
class_prediction_svm = class_prediction_svm.reshape(img[:, :, 0].shape)
save_classification_map(f'SVM_classification_map_{subset_size}.tif', class_prediction_svm, meta)
print (f'SVM prediction map saved as SVM_classification_map.tif')
plot_confusion_matrix(svm_conf_matrix, f'SVM Confusion Matrix', f'SVM_confusion_matrix_{subset_size}.png', classes)
end_time_SVM = time.time()
SVM_time = end_time_SVM - start_time2
print(f'Time consumed: {SVM_time} seconds')

# Save results
results = {
    'Random Forest': {
        'Best Parameters': best_rf,
        'Cross-validation Accuracy': rf_trials.best_trial['result'].get('loss', 'N/A'),
        'Testing Accuracy': rf_accuracy_test,
        'Testing Kappa': rf_kappa_test,
        'Random Forest f1': rf_f1,
        'Random Forest recall': rf_recall,
        'Random Forest precision': rf_precision,
        'Time consuming': RF_time
    },
    'SVM': {
        'Best Parameters': best_svm,
        'Cross-validation Accuracy': svm_trials.best_trial['result'].get('loss', 'N/A'),
        'Testing Accuracy': svm_accuracy_test,
        'Testing Kappa': svm_kappa_test,
        'SVM f1': svm_f1,
        'SVM recall': svm_recall,
        'SVM precision': svm_precision,
        'Time consuming': SVM_time
    }
}

with open(f'model_results_{subset_size}.txt', 'w') as f:
    for model, data in results.items():
        f.write(f"{model}:\n")
        for key, value in data.items():
            f.write(f"  {key}: {value}\n")

with open(f'model_results_{subset_size}.json', 'w') as f:
    json.dump(results, f, indent=4)


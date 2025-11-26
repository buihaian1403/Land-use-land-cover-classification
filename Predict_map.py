import torch
import pandas as pd
import numpy as np
import gc
from torch.utils.data import DataLoader
from torchvision import transforms
from models.soc.soc import SoC
from utils import net_builder, get_logger
from imblearn.over_sampling import SMOTE
# Define transform
transform = transforms.Lambda(lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).float())

# Load data
def load_data(inputNPZ, label_csv):
    with np.load(inputNPZ, allow_pickle=False) as npz_file:
        data = npz_file["data"]
        del npz_file
        gc.collect()

    for band in range(data.shape[-1]):
        band_data = data[:, :, :, band]
        valid_mask = band_data != -32768 & np.isnan(band_data)
        mean_val = np.mean(band_data[valid_mask])
        band_data[~valid_mask] = mean_val
        data[:, :, :, band] = band_data

    ijIncl = pd.read_csv(label_csv)
    labels = ijIncl.iloc[:, 2].values
    labeled_mask = labels != 0
    unlabeled_mask = labels == 0

    data_labeled = data[labeled_mask]
    labels_labeled = labels[labeled_mask]
    data_unlabeled = data[unlabeled_mask]
    unlabeled_indices = np.where(unlabeled_mask)[0]

    return ijIncl, data_unlabeled, data_labeled, labels_labeled, unlabeled_indices

def balance_labeled_data(data_labeled, labels_labeled):
    rus = SMOTE(random_state=42)
    data_labeled, labels_labeled = rus.fit_resample(data_labeled.reshape(data_labeled.shape[0], -1), labels_labeled)
    data_labeled = data_labeled.reshape(-1, 24, 24, 10)
    return data_labeled, labels_labeled
# Get model checkpoint path
def get_model_path_for_chunk(chunk_idx, model_base_path):
    return f"{model_base_path}/latest_model_chunk_{chunk_idx}.pth"

# Save predictions to CSV
def update_predictions_in_csv(ijIncl, y_pred, unlabeled_indices, output_csv):
    ijIncl.loc[unlabeled_indices, 'label'] = y_pred
    ijIncl.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Predict with saved models
def predict_with_saved_models(model_base_path, data_unlabeled, unlabeled_indices, chunk_size, batch_size, args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    y_pred_full = []  # Global predictions

    # Prepare model
    _net_builder = net_builder(
        args.net,
        {
            "depth": args.depth,
            "widen_factor": args.widen_factor,
            "leaky_slope": args.leaky_slope,
            "bn_momentum": 1.0 - args.ema_m,
            "dropRate": args.dropout,
        },
    )
    model = SoC(
        _net_builder,
        args.num_classes,
        args.ema_m,
        args.ulb_loss_ratio,
        num_eval_iter=args.num_eval_iter,
        num_train_iter=args.num_train_iter,
        num_tracked_batch=args.num_tracked_batch,
        alpha=args.alpha,
        save_dir=args.save_dir,
        save_name=args.save_name,
        gpu=args.gpu,
    )
    model.eval_model = model.eval_model.to(device)

    # Predict in chunks
    for chunk_idx in range(0, len(data_unlabeled), chunk_size):
        # Ensure the slicing does not exceed the array bounds
        chunk_start = chunk_idx
        chunk_end = min(chunk_idx + chunk_size, len(data_unlabeled))
        chunk_data = data_unlabeled[chunk_start:chunk_end]
        chunk_indices = unlabeled_indices[chunk_start:chunk_end]
        chunk_tensor = transform(chunk_data)

        model_path = get_model_path_for_chunk(chunk_idx, model_base_path)
        checkpoint = torch.load(model_path, map_location=device)
        model.eval_model.load_state_dict(checkpoint["eval_model"])
        model.eval_model.eval()

        chunk_predictions = []
        with torch.no_grad():
            for i in range(0, len(chunk_tensor), batch_size):
                batch = chunk_tensor[i:i + batch_size].to(device)
                outputs = model.eval_model(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs, 1)
                chunk_predictions.extend(predicted.cpu().numpy())
        y_pred_full.extend(chunk_predictions)
    return np.array(y_pred_full)

# Main function
def main(inputNPZ, label_csv, model_base_path, output_csv, chunk_size=100000, batch_size=128, args=None):
    # Load data
    ijIncl, data_unlabeled, data_labeled, labels_labeled, unlabeled_indices = load_data(inputNPZ, label_csv)
    data_labeled, labels_labeled = balance_labeled_data(data_labeled, labels_labeled)
    # Predict
    y_pred = predict_with_saved_models(
        model_base_path, data_unlabeled, unlabeled_indices, chunk_size, batch_size, args
    )

    # Save predictions
    update_predictions_in_csv(ijIncl, y_pred, unlabeled_indices, output_csv)

# Run script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="cnn13")
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--ema_m", type=float, default=0.999)
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--save_name", type=str, default="soc")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--num_train_iter', type=int, default=2000, 
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=2000,
                    help='evaluation frequency')
    parser.add_argument('--num_tracked_batch', type=int, default=32, help='total number of batch tracked by CTT')
    parser.add_argument('--alpha', type=float, default=1.1, help='use {2.5,4} for {semi_aves,semi_fungi}')
    args = parser.parse_args()

    inputNPZ = "HighRes_data_tiles_2023.npz"
    label_csv = "HighRes_data_tiles_2023_ij_included.csv"
    model_base_path = "saved_models/soc"
    output_csv = "Soc_prediction.csv"

    main(inputNPZ, label_csv, model_base_path, output_csv, args=args)

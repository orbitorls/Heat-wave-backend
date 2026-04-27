import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from heatwave_model import HeatwaveConvLSTM, PhysicsInformedLoss
from data_loader import load_era5_data, create_sequences, clean_data, normalize_data

# --- CONFIG ---
DATA_DIR = "era5_data"
MODELS_DIR = "models"
BATCH_SIZE = 4
SEQ_LEN = 5
FUTURE_SEQ = 2
EPOCHS = 30
LEARNING_RATE = 1e-4
CHANNELS = 8  # Z, T2M, SWVL1, TP, HUMIDITY, ELEV, LAT, LON
HIDDEN_DIM = [32, 32]
KERNEL_SIZE = [(3, 3), (3, 3)]
NUM_LAYERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_next_version(model_dir: str) -> int:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1
    import glob
    files = glob.glob(os.path.join(model_dir, "heatwave_convlstm_v*.pth"))
    if not files: return 1
    versions = [int(f.split("_v")[-1].split(".")[0]) for f in files]
    return max(versions) + 1

def train():
    print(f"🚀 Starting ConvLSTM Training for Thailand (Device: {device})")
    
    # 1. Load Data
    print("[1/6] Loading ERA5 Data with Spatial Features...")
    data_raw, lats, lons, _, _ = load_era5_data(DATA_DIR, normalize=False)
    
    # 2. Clean data (remove outliers before normalization)
    print("[2/6] Cleaning data...")
    data_clean, clip_bounds = clean_data(data_raw)
    
    # 3. Temporal split (70/15/15) BEFORE normalization to prevent leakage
    print("[3/6] Temporal split (70/15/15)...")
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    n = len(data_clean)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_data = data_clean[:train_end]
    val_data = data_clean[train_end:val_end]
    test_data = data_clean[val_end:]
    print(f"      Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 4. Compute normalization on TRAINING split only
    print("[4/6] Computing normalization stats (training split only)...")
    mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
    std = train_data.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    
    train_norm = normalize_data(train_data, mean, std)
    val_norm = normalize_data(val_data, mean, std)
    test_norm = normalize_data(test_data, mean, std)
    
    # 5. Create Sequences
    print("[5/6] Creating sequences...")
    train_x, train_y = create_sequences(train_norm, seq_len=SEQ_LEN, pred_len=FUTURE_SEQ)
    val_x, val_y = create_sequences(val_norm, seq_len=SEQ_LEN, pred_len=FUTURE_SEQ)
    test_x, test_y = create_sequences(test_norm, seq_len=SEQ_LEN, pred_len=FUTURE_SEQ)
    
    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    val_ds = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 6. Initialize Model & Optimizer
    model = HeatwaveConvLSTM(
        input_dim=CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    criterion = PhysicsInformedLoss(lambda_phy=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"[6/6] Training for {EPOCHS} epochs...")
    save_path = None
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x, future_seq=FUTURE_SEQ)
            loss, mse, phy = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x, future_seq=FUTURE_SEQ)
                loss, _, _ = criterion(output, batch_y)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            v = get_next_version(MODELS_DIR)
            save_path = os.path.join(MODELS_DIR, f"heatwave_convlstm_v{v}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
                'normalization_mean': mean.tolist() if hasattr(mean, 'tolist') else mean,
                'normalization_std': std.tolist() if hasattr(std, 'tolist') else std,
                'clip_bounds': clip_bounds,
                'input_dim': CHANNELS,
                'hidden_dim': HIDDEN_DIM,
                'kernel_size': KERNEL_SIZE,
                'num_layers': NUM_LAYERS,
                'seq_len': SEQ_LEN,
                'future_seq': FUTURE_SEQ,
                'lats': lats.tolist() if hasattr(lats, 'tolist') else lats,
                'lons': lons.tolist() if hasattr(lons, 'tolist') else lons,
                'channels': CHANNELS,
                'model_type': 'convlstm',
                'metadata': {
                    'model_type': 'convlstm',
                    'input_dim': CHANNELS,
                    'hidden_dim': HIDDEN_DIM,
                    'kernel_size': KERNEL_SIZE,
                    'num_layers': NUM_LAYERS,
                    'normalization_mean': mean.tolist() if hasattr(mean, 'tolist') else mean,
                    'normalization_std': std.tolist() if hasattr(std, 'tolist') else std,
                    'clip_bounds': clip_bounds,
                    'seq_len': SEQ_LEN,
                    'future_seq': FUTURE_SEQ,
                }
            }, save_path)
            # Remove old versions to save space
            # (Optional)
    
    print(f"✅ Training Complete. Best model saved to: {save_path or 'N/A'}")

if __name__ == "__main__":
    train()

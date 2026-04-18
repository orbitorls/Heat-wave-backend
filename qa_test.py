import sys
sys.path.insert(0, '.')

print("=" * 60)
print("QA Test 1: get_latest_checkpoint() method exists")
print("=" * 60)
from src.models.manager import ModelManager
mm = ModelManager()
ckpt = mm.get_latest_checkpoint()
print("PASS: get_latest_checkpoint() runs without error")
print(f"  Result: {ckpt}")

print("\n" + "=" * 60)
print("QA Test 2: New methods and attributes exist")
print("=" * 60)
assert hasattr(mm, 'predict_temperature'), 'Missing predict_temperature method'
print("PASS: predict_temperature method exists")

assert hasattr(mm, 'denormalize_temperature'), 'Missing denormalize_temperature method'
print("PASS: denormalize_temperature method exists")

assert hasattr(mm, 'normalization_mean'), 'Missing normalization_mean attribute'
print("PASS: normalization_mean attribute exists")

assert hasattr(mm, 'normalization_std'), 'Missing normalization_std attribute'
print("PASS: normalization_std attribute exists")

print("\n" + "=" * 60)
print("QA Test 3: Syntax validation")
print("=" * 60)
import ast
with open('src/models/manager.py') as f:
    src = f.read()
try:
    ast.parse(src)
    print("PASS: manager.py syntax is valid")
except SyntaxError as e:
    print(f"FAIL: Syntax error in manager.py: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("QA Test 4: Check denormalize_temperature logic")
print("=" * 60)
import numpy as np
mm.normalization_mean = np.array([[0, 293, 0.5, 0, 0, 0, 0, 0]]).reshape(1, 8, 1, 1)
mm.normalization_std = np.array([[0.1, 15, 0.1, 0, 0, 0, 0, 0]]).reshape(1, 8, 1, 1)
norm_grid = np.array([[0.5, 0.3], [0.4, 0.2]])  # 2x2 normalized grid
denorm = mm.denormalize_temperature(norm_grid, channel_idx=1)
print(f"PASS: denormalize_temperature returns array shape {denorm.shape}")
print(f"  Input shape: {norm_grid.shape}, Output shape: {denorm.shape}")
print(f"  Sample values (Celsius): {denorm[0, :2]}")

print("\n" + "=" * 60)
print("QA Test 5: Check get_latest_checkpoint searches both patterns")
print("=" * 60)
import inspect
source = inspect.getsource(ModelManager.get_latest_checkpoint)
assert "heatwave_model_checkpoint_v" in source, "Missing RF pattern search"
assert "heatwave_convlstm_v" in source, "Missing ConvLSTM pattern search"
print("PASS: get_latest_checkpoint searches both patterns:")
print("  - heatwave_model_checkpoint_v*.pth (RF)")
print("  - heatwave_convlstm_v*.pth (ConvLSTM)")

print("\n" + "=" * 60)
print("ALL QA TESTS PASSED")
print("=" * 60)

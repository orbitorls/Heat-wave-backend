#!/usr/bin/env python3
import os

# Check the last 50 lines of Paper.log
with open(r"D:\Heat-wave-backend\Paper.log", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
    print("LAST 50 LINES OF Paper.log:")
    print("=" * 80)
    for line in lines[-50:]:
        print(line, end="")
    
print("\n" + "=" * 80)

# Check file sizes
print("\nFILE ARTIFACTS:")
print("=" * 80)
files_to_check = ["Paper.pdf", "Paper.aux", "Paper.log", "Paper.out"]
for fname in files_to_check:
    fpath = os.path.join(r"D:\Heat-wave-backend", fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        mtime = os.path.getmtime(fpath)
        from datetime import datetime
        mod_time = datetime.fromtimestamp(mtime)
        print(f"✓ {fname}: {size:,} bytes (modified: {mod_time})")
    else:
        print(f"✗ {fname}: NOT FOUND")

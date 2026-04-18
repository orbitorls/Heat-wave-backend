import subprocess
import os

os.chdir(r"D:\Heat-wave-backend")

print("=" * 80)
print("COMPILATION STATUS AND PDF VERIFICATION")
print("=" * 80)

# Read and show last 50 lines of log
with open("Paper.log", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
    print("\nLAST 50 LINES OF Paper.log:")
    print("-" * 80)
    for line in lines[-50:]:
        print(line, end="")

print("\n" + "=" * 80)
print("FILE ARTIFACTS:")
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

print("\n" + "=" * 80)
print("ATTEMPTING NEW COMPILATION RUNS...")
print("=" * 80)

# First run
print("\nFIRST RUN:")
print("-" * 80)
try:
    result1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "Paper.tex"],
        capture_output=True,
        text=True,
        timeout=120
    )
    # Show last 2000 chars of output
    stdout_to_show = result1.stdout[-2000:] if len(result1.stdout) > 2000 else result1.stdout
    print("Last 2000 chars of STDOUT:")
    print(stdout_to_show)
    if result1.stderr:
        print("\nSTDERR:")
        stderr_to_show = result1.stderr[-1000:] if len(result1.stderr) > 1000 else result1.stderr
        print(stderr_to_show)
    print(f"\nReturn code: {result1.returncode}")
except FileNotFoundError as e:
    print(f"ERROR: pdflatex not found - {e}")
    print("Trying with latexmk instead...")
    try:
        result1 = subprocess.run(
            ["latexmk", "-pdf", "Paper.tex"],
            capture_output=True,
            text=True,
            timeout=120
        )
        stdout_to_show = result1.stdout[-2000:] if len(result1.stdout) > 2000 else result1.stdout
        print("Last 2000 chars of STDOUT:")
        print(stdout_to_show)
        if result1.stderr:
            print("\nSTDERR:")
            stderr_to_show = result1.stderr[-1000:] if len(result1.stderr) > 1000 else result1.stderr
            print(stderr_to_show)
        print(f"\nReturn code: {result1.returncode}")
    except FileNotFoundError:
        print("ERROR: latexmk also not found")

# Second run
print("\n\nSECOND RUN (for cross-references):")
print("-" * 80)
try:
    result2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "Paper.tex"],
        capture_output=True,
        text=True,
        timeout=120
    )
    stdout_to_show = result2.stdout[-2000:] if len(result2.stdout) > 2000 else result2.stdout
    print("Last 2000 chars of STDOUT:")
    print(stdout_to_show)
    if result2.stderr:
        print("\nSTDERR:")
        stderr_to_show = result2.stderr[-1000:] if len(result2.stderr) > 1000 else result2.stderr
        print(stderr_to_show)
    print(f"\nReturn code: {result2.returncode}")
except FileNotFoundError as e:
    print(f"ERROR: pdflatex not found - {e}")
    try:
        result2 = subprocess.run(
            ["latexmk", "-pdf", "Paper.tex"],
            capture_output=True,
            text=True,
            timeout=120
        )
        stdout_to_show = result2.stdout[-2000:] if len(result2.stdout) > 2000 else result2.stdout
        print("Last 2000 chars of STDOUT:")
        print(stdout_to_show)
        if result2.stderr:
            print("\nSTDERR:")
            stderr_to_show = result2.stderr[-1000:] if len(result2.stderr) > 1000 else result2.stderr
            print(stderr_to_show)
        print(f"\nReturn code: {result2.returncode}")
    except FileNotFoundError:
        print("ERROR: latexmk also not found")

print("\n" + "=" * 80)
print("FINAL PDF STATUS:")
print("=" * 80)
pdf_path = r"D:\Heat-wave-backend\Paper.pdf"
if os.path.exists(pdf_path):
    size = os.path.getsize(pdf_path)
    print(f"✓ SUCCESS: Paper.pdf exists")
    print(f"  File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
else:
    print(f"✗ FAILED: Paper.pdf does not exist")

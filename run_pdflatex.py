#!/usr/bin/env python3
import subprocess
import os
import sys

os.chdir(r"D:\Heat-wave-backend")

print("=" * 80)
print("FIRST RUN - Compiling Paper.tex with pdflatex")
print("=" * 80)

try:
    result1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "Paper.tex"],
        capture_output=True,
        text=True,
        timeout=120
    )
    print("STDOUT (last 3000 chars):")
    print(result1.stdout[-3000:] if len(result1.stdout) > 3000 else result1.stdout)
    print("\nSTDERR (last 1000 chars):")
    print(result1.stderr[-1000:] if len(result1.stderr) > 1000 else result1.stderr)
    print("\nReturn code:", result1.returncode)
except FileNotFoundError:
    print("ERROR: pdflatex not found in PATH")
    print("Trying with full MiKTeX path...")
    try:
        result1 = subprocess.run(
            [r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe", "-interaction=nonstopmode", "Paper.tex"],
            capture_output=True,
            text=True,
            timeout=120
        )
        print("STDOUT (last 3000 chars):")
        print(result1.stdout[-3000:] if len(result1.stdout) > 3000 else result1.stdout)
        print("\nSTDERR (last 1000 chars):")
        print(result1.stderr[-1000:] if len(result1.stderr) > 1000 else result1.stderr)
        print("\nReturn code:", result1.returncode)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

print("\n" + "=" * 80)
print("SECOND RUN - Compiling again for cross-references")
print("=" * 80)

try:
    result2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "Paper.tex"],
        capture_output=True,
        text=True,
        timeout=120
    )
    print("STDOUT (last 3000 chars):")
    print(result2.stdout[-3000:] if len(result2.stdout) > 3000 else result2.stdout)
    print("\nSTDERR (last 1000 chars):")
    print(result2.stderr[-1000:] if len(result2.stderr) > 1000 else result2.stderr)
    print("\nReturn code:", result2.returncode)
except FileNotFoundError:
    print("ERROR: pdflatex not found in PATH")
    print("Trying with full MiKTeX path...")
    try:
        result2 = subprocess.run(
            [r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe", "-interaction=nonstopmode", "Paper.tex"],
            capture_output=True,
            text=True,
            timeout=120
        )
        print("STDOUT (last 3000 chars):")
        print(result2.stdout[-3000:] if len(result2.stdout) > 3000 else result2.stdout)
        print("\nSTDERR (last 1000 chars):")
        print(result2.stderr[-1000:] if len(result2.stderr) > 1000 else result2.stderr)
        print("\nReturn code:", result2.returncode)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

print("\n" + "=" * 80)
print("CHECKING FOR OUTPUT PDF")
print("=" * 80)
pdf_path = r"D:\Heat-wave-backend\Paper.pdf"
if os.path.exists(pdf_path):
    file_size = os.path.getsize(pdf_path)
    print(f"✓ SUCCESS: Paper.pdf exists")
    print(f"  Full path: {pdf_path}")
    print(f"  File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
else:
    print(f"✗ FAILED: Paper.pdf does not exist at {pdf_path}")

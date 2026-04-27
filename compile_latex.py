import subprocess, os, sys

os.chdir(r"D:\Heat-wave-backend")

print("=" * 80)
print("FIRST RUN - Compiling Paper.tex with pdflatex")
print("=" * 80)
result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "Paper.tex"],
    capture_output=True, text=True, cwd=r"D:\Heat-wave-backend"
)
print("STDOUT (last 3000 chars):")
print(result.stdout[-3000:])
print("\nSTDERR (last 1000 chars):")
print(result.stderr[-1000:])
print("\nReturn code:", result.returncode)

print("\n" + "=" * 80)
print("SECOND RUN - Compiling again for cross-references")
print("=" * 80)
result2 = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "Paper.tex"],
    capture_output=True, text=True, cwd=r"D:\Heat-wave-backend"
)
print("STDOUT (last 3000 chars):")
print(result2.stdout[-3000:])
print("\nSTDERR (last 1000 chars):")
print(result2.stderr[-1000:])
print("\nReturn code:", result2.returncode)

print("\n" + "=" * 80)
print("CHECKING FOR OUTPUT PDF")
print("=" * 80)
pdf_path = r"D:\Heat-wave-backend\Paper.pdf"
if os.path.exists(pdf_path):
    print(f"✓ SUCCESS: {pdf_path} exists")
    file_size = os.path.getsize(pdf_path)
    print(f"  File size: {file_size} bytes")
else:
    print(f"✗ FAILED: {pdf_path} does not exist")

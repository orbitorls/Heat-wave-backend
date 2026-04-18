import subprocess, os, sys

os.chdir(r"D:\Heat-wave-backend")
pdflatex = r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"

for run in range(1, 3):
    print(f"--- Pass {run}/2 ---")
    result = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "Paper.tex"],
        capture_output=True, text=True, cwd=r"D:\Heat-wave-backend"
    )
    print("Return code:", result.returncode)
    if result.returncode != 0:
        print("STDOUT (last 2000):", result.stdout[-2000:])
        print("STDERR:", result.stderr[-500:])
        sys.exit(1)

print("SUCCESS: Paper.pdf compiled.")
print("Exists:", os.path.exists(r"D:\Heat-wave-backend\Paper.pdf"))

import re
from pathlib import Path

screens_dir = Path(r'D:\Heat-wave-backend\src\tui\screens')

for f in screens_dir.glob('*.py'):
    if f.name == '__init__.py':
        continue
    content = f.read_text()
    ids = re.findall(r'id="([^"]+)"', content)
    seen = set()
    dups = set()
    for id_ in ids:
        if id_ in seen:
            dups.add(id_)
        seen.add(id_)
    if dups:
        print(f"{f.name}: duplicate IDs = {dups}")
    else:
        print(f"{f.name}: OK")

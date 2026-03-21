from pathlib import Path

from src.app.paths import AppPaths

paths = AppPaths(base_dir=Path("storage"))
paths.ensure_exists()

print(paths.documents_dir)
print(paths.metadata_dir)
print(paths.qdrant_dir)
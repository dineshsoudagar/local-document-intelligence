"""Print the configured application storage paths."""

from pathlib import Path

from src.app.paths import AppPaths


def main() -> None:
    """Create storage paths and print their resolved locations."""
    paths = AppPaths(base_dir=Path("storage"))
    paths.ensure_exists()

    print(paths.documents_dir)
    print(paths.metadata_dir)
    print(paths.qdrant_dir)


if __name__ == "__main__":
    main()

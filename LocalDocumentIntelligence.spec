# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = Path(SPECPATH).resolve()
hiddenimports = collect_submodules("webview")
excluded_runtime_packages = [
    "bitsandbytes",
    "cv2",
    "docling",
    "docling_core",
    "docling_parse",
    "fastembed",
    "llvmlite",
    "nltk",
    "numba",
    "onnxruntime",
    "pandas",
    "pypdfium2",
    "pypdfium2_raw",
    "qdrant_client",
    "rapidocr",
    "scipy",
    "sentence_transformers",
    "sklearn",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
]


a = Analysis(
    [str(project_root / "launcher" / "desktop_launcher.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "bundle"), "bundle"),
        (str(project_root / "frontend" / "dist"), "frontend/dist"),
        (str(project_root / "scripts"), "scripts"),
        (str(project_root / "src"), "src"),
        (str(project_root / "requirements.txt"), "."),
        (str(project_root / "requirements-launcher.txt"), "."),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_runtime_packages,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalDocumentIntelligence",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LocalDocumentIntelligence",
)

# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata


project_root = Path(SPECPATH).resolve()
hiddenimports = collect_submodules("webview")
metadata_datas = []
for distribution_name in (
    "docling",
    "docling-core",
    "docling-ibm-models",
    "docling-parse",
):
    metadata_datas += copy_metadata(distribution_name)


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
    ] + metadata_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
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

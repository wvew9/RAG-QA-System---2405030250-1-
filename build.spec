# -*- mode: python ; coding: utf-8 -*-

import streamlit
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect Streamlit data files
streamlit_data = collect_data_files('streamlit')

# Collect hidden imports
hidden_imports = [
    'streamlit',
    'streamlit.runtime',
    'chromadb',
    'langchain',
    'langchain_community',
    'langchain_core',
    'langchain_chroma',
    'PyPDF2',
    'docx',
    'tiktoken',
]
hidden_imports += collect_submodules('streamlit')
hidden_imports += collect_submodules('chromadb')
hidden_imports += collect_submodules('langchain')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=streamlit_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RAG-QA-System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

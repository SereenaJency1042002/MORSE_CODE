# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# customtkinter needs its theme JSON files and images bundled
tmp = collect_all('customtkinter')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# sounddevice needs the PortAudio DLL
tmp = collect_all('sounddevice')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# soundfile needs the libsndfile DLL
tmp = collect_all('soundfile')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# librosa data files + submodules (heavy but required)
tmp = collect_all('librosa')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# sklearn – only KMeans is used but Cython extensions need binaries collected
tmp = collect_all('sklearn')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# matplotlib backend used by the embedded graph
hiddenimports += ['matplotlib.backends.backend_tkagg']

# scipy submodules used by signal_filter.py
hiddenimports += [
    'scipy.signal',
    'scipy.signal.windows',
    'scipy._lib.messagestream',
    'scipy.special._ufuncs_cxx',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
    'scipy.sparse.csgraph._validation',
]

# src package: ui_display.py imports these lazily inside functions,
# so PyInstaller cannot detect them during analysis — must be explicit.
hiddenimports += [
    'src.audio_loader',
    'src.signal_filter',
    'src.signal_visualizer',
    'src.morse_decoder',
    'src.intelligent_corrector',
    'src.ai_predictor',
    'src.ui_display',
    'groq',
    'dotenv',
]

# Include config.json and the src package folder
datas += [('config.json', '.')]
datas += [('src', 'src')]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
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
    [],
    exclude_binaries=True,
    name='MorseDecoder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    name='MorseDecoder',
)

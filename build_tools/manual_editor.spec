# PyInstaller spec — manual_editor 打包成单文件 exe
# 用法（在项目根，mitsubishi 环境）：
#   python -m PyInstaller build_tools/manual_editor.spec --noconfirm
# 产物：dist/ManualEditor.exe（客户双击即用，无需 Python 环境）

block_cipher = None

a = Analysis(
    ['../manual_editor/main.py'],
    pathex=['..'],                      # 项目根加入路径，能 import manual_editor 包
    binaries=[],
    datas=[
        # 字体打进 exe，运行时解压到 sys._MEIPASS/fonts/（font_config 已支持）
        ('../fonts/basictitlefont-1.ttf', 'fonts'),
        # logo 打进 exe，运行时解压到 sys._MEIPASS/assets/（_find_logo 已支持窗口图标）
        ('../manual_editor/assets/logo.ico', 'assets'),
        ('../manual_editor/assets/logo.png', 'assets'),
    ],
    hiddenimports=[
        'manual_editor',
        'manual_editor.app',
        'manual_editor.app.main_window',
        'manual_editor.app.box_item',
        'manual_editor.app.csv_loader',
        'manual_editor.app.editor_view',
        'manual_editor.app.filename_map',
        'manual_editor.app.font_config',
        'manual_editor.app.original_view',
        'manual_editor.app.render',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 排除无关大依赖，减小体积（manual_editor 不用这些）
        'paddle', 'paddlex', 'paddleocr', 'cv2', 'torch',
        'fastapi', 'uvicorn', 'gradio',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ManualEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,                      # GUI 程序，无控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../manual_editor/assets/logo.ico',   # exe 文件图标
)

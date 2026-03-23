import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import cv2
from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions

TIF_DIR = Path(r'c:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo')
files = [
    TIF_DIR / 'YA050C055_CD.tif',
    TIF_DIR / 'YA097C376$1_D.tif',
    TIF_DIR / 'YA026D941_0d.tif',
]

for f in files:
    print(f'\n=== {f.name} ===')
    img, _ = load_file(str(f))
    regions = detect_all_regions(img, prefixes=['X','Y','Z','B'])
    rot = regions.get('_metadata',{}).get('rotation')
    if rot is not None:
        img = cv2.rotate(img, rot)
    orange = regions.get('top_left_number')
    meta = regions.get('_metadata', {})
    print(f'  orange bbox: {orange}')
    print(f'  orange text: {meta.get("top_left_text")}')
    green = regions.get('bottom_right_number')
    print(f'  green bbox: {green}')
    print(f'  green text: {meta.get("bottom_right_text")}')

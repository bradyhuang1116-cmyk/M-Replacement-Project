import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import cv2
from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions, _enhance_vertical_lines

TIF_DIR = Path(r'c:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo')
f = TIF_DIR / 'YA050C055_CD.tif'

img_array, _ = load_file(str(f))
enhanced = _enhance_vertical_lines(img_array)
regions = detect_all_regions(enhanced, prefixes=['X','Y','Z','B'])

rot = regions.get('_metadata',{}).get('rotation')
if rot is not None:
    img_array = cv2.rotate(img_array, rot)

meta = regions.get('_metadata', {})
print(f'orange bbox: {regions.get("top_left_number")}')
print(f'orange text: {meta.get("top_left_text")}')
print(f'green bbox: {regions.get("bottom_right_number")}')
print(f'green text: {meta.get("bottom_right_text")}')
print(f'red bbox: {regions.get("material_code_column")}')

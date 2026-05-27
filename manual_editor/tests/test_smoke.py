"""无界面 smoke 测试：模块可 import + render 能写出图片 + csv_loader 能读 CSV。"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def test_imports_all_modules():
    # 仅 import，不实例化 QApplication
    from manual_editor.app import (
        box_item,
        csv_loader,
        editor_view,
        filename_map,
        font_config,
        main_window,
        original_view,
        render,
    )
    assert box_item.BoxState.LOCKED
    assert hasattr(render, "render_committed_boxes")


def test_render_committed_boxes(tmp_path: Path):
    from manual_editor.app.render import CommittedBox, render_committed_boxes

    src = tmp_path / "src.tif"
    Image.new("RGB", (400, 200), (128, 128, 128)).save(src)

    out = tmp_path / "out.tif"
    render_committed_boxes(
        src,
        [CommittedBox(50, 50, 200, 60, "YA999")],
        out,
    )

    assert out.exists()
    img = Image.open(out)
    assert img.size == (400, 200)
    # 框中心应是白底（非灰）
    assert img.getpixel((150, 80)) == (255, 255, 255)


def test_csv_loader_roundtrip(tmp_path: Path):
    from manual_editor.app.csv_loader import load_csv

    csv_path = tmp_path / "y_boxes.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "token", "x1", "y1", "x2", "y2"])
        w.writerow(["XXX", "YA001", "10", "20", "110", "70"])
        w.writerow(["XXX", "YA002", "200", "300", "260", "330"])
        w.writerow(["YYY", "B777", "0", "0", "50", "50"])

    out = load_csv(tmp_path)
    assert set(out.keys()) == {"XXX", "YYY"}
    assert len(out["XXX"]) == 2
    rec = out["XXX"][0]
    assert rec.token == "YA001"
    assert rec.w == 100 and rec.h == 50


def test_filename_to_csv_lookup():
    """模拟完整链路：HXXX-R.tif -> stem -> 在 CSV dict 里查到记录。"""
    from manual_editor.app.filename_map import replaced_to_stem

    csv_data = {"YA116A226-1_Eg-脱敏": ["dummy"]}
    stem = replaced_to_stem("HYA116A226-1_Eg-脱敏-R.tif")
    assert stem in csv_data


def test_csv_loader_handles_both_source_file_formats(tmp_path: Path):
    """旧 factory_note_pixel_v6 写 'XXX.tif'，新 text_replacer 写 'XXX' —— 两种都要能查到。"""
    from manual_editor.app.csv_loader import load_csv
    from manual_editor.app.filename_map import replaced_to_stem

    csv_path = tmp_path / "y_boxes.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "token", "x1", "y1", "x2", "y2"])
        w.writerow(["YA057C857_0-脱敏.tif", "YA057C800", "10", "20", "60", "70"])  # 旧
        w.writerow(["B507795_C-脱敏", "B507795", "100", "200", "300", "400"])      # 新
    out = load_csv(tmp_path)
    # 两种都应该归一为裸 stem
    assert set(out.keys()) == {"YA057C857_0-脱敏", "B507795_C-脱敏"}
    # 用 replaced_to_stem 反推：HXXX-R.tif -> 命中
    assert replaced_to_stem("HYA057C857_0-脱敏-R.tif") in out
    assert replaced_to_stem("HB507795_C-脱敏-R.tif") in out

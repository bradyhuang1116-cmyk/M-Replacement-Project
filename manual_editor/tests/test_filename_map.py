"""filename_map 单元测试。"""
from __future__ import annotations

import pytest

from manual_editor.app.filename_map import (
    is_replaced_name,
    replaced_to_original_name,
    replaced_to_stem,
)


class TestReplacedToStem:
    def test_typical(self):
        assert replaced_to_stem("HYA116A226-1_Eg-脱敏-R.tif") == "YA116A226-1_Eg-脱敏"

    def test_tiff_extension(self):
        assert replaced_to_stem("HB507795_C-脱敏-R.tiff") == "B507795_C-脱敏"

    def test_uppercase_extension(self):
        assert replaced_to_stem("HP101081A110-1_h-脱敏-R.TIF") == "P101081A110-1_h-脱敏"

    def test_no_h_prefix(self):
        # 不符合规则时，按尽力剥离原则
        assert replaced_to_stem("XYZ-R.tif") == "XYZ"

    def test_no_r_suffix(self):
        assert replaced_to_stem("HXYZ.tif") == "XYZ"

    def test_no_extension_returns_stem(self):
        assert replaced_to_stem("HYA116-R") == "YA116"


class TestReplacedToOriginalName:
    def test_tif(self):
        assert replaced_to_original_name("HYA116A226-1_Eg-脱敏-R.tif") == "YA116A226-1_Eg-脱敏.tif"

    def test_tiff(self):
        assert replaced_to_original_name("HB507795_C-脱敏-R.tiff") == "B507795_C-脱敏.tiff"

    def test_uppercase_preserves_casing(self):
        # 保留原扩展名大小写（case-sensitive FS 下需要精确匹配）
        assert replaced_to_original_name("HXYZ-R.TIF") == "XYZ.TIF"

    def test_unknown_ext_defaults_to_tif(self):
        assert replaced_to_original_name("HXYZ-R.png") == "XYZ.tif"


class TestIsReplacedName:
    @pytest.mark.parametrize("name", [
        "HYA116A226-1_Eg-脱敏-R.tif",
        "HB507795_C-脱敏-R.tiff",
        "HXYZ-R.TIF",
    ])
    def test_valid(self, name):
        assert is_replaced_name(name) is True

    @pytest.mark.parametrize("name", [
        "YA116A226-1_Eg-脱敏.tif",  # 缺 H 前缀
        "HXYZ.tif",                  # 缺 -R 后缀
        "HXYZ-R.png",                # 非 tif/tiff
        "HXYZ-R",                    # 无扩展名
        "edited",                    # 子目录名
    ])
    def test_invalid(self, name):
        assert is_replaced_name(name) is False

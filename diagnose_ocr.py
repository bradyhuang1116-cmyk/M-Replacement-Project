"""OCR环境诊断脚本 - 检查PaddleOCR配置和模型"""
import sys
import platform

print("=" * 60)
print("系统信息")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"平台: {platform.platform()}")
print(f"处理器: {platform.processor()}")
print()

print("=" * 60)
print("检查依赖库")
print("=" * 60)

try:
    import paddle
    print(f"✓ PaddlePaddle: {paddle.__version__}")
    print(f"  设备: {paddle.device.get_device()}")
except Exception as e:
    print(f"✗ PaddlePaddle 导入失败: {e}")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
    print(f"✓ PaddleOCR 已安装")
except Exception as e:
    print(f"✗ PaddleOCR 导入失败: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV 导入失败: {e}")

print()
print("=" * 60)
print("测试 OCR 初始化")
print("=" * 60)

configs = [
    ("CPU + Mobile", {"use_gpu": False, "use_angle_cls": True, "lang": "ch",
                      "det_model_dir": None, "rec_model_dir": None}),
    ("CPU + Server", {"use_gpu": False, "use_angle_cls": True, "lang": "ch",
                      "det_model_dir": None, "rec_model_dir": None,
                      "det_db_box_thresh": 0.3, "det_db_unclip_ratio": 1.6})
]

for name, config in configs:
    print(f"\n测试: {name}")
    try:
        ocr = PaddleOCR(**config)
        print(f"  ✓ 初始化成功")

        # 测试简单识别
        import numpy as np
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        result = ocr.ocr(test_img, cls=True)
        print(f"  ✓ 识别测试通过 (结果: {len(result[0]) if result and result[0] else 0} 个)")

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

print()
print("=" * 60)
print("诊断完成")
print("=" * 60)

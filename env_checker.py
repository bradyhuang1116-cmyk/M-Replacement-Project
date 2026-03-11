"""环境检查和诊断工具"""
import sys
import importlib

REQUIRED_PACKAGES = {
    "paddleocr": "2.7.0",
    "paddlepaddle": "2.5.0",
    "fastapi": "0.100.0",
    "gradio": "4.0.0",
    "cv2": "4.8.0",
    "PIL": "10.0.0",
    "numpy": "1.24.0",
}

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and 11 <= version.minor <= 12:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    return False, f"Python {version.major}.{version.minor}.{version.micro} ✗ (需要 3.11-3.12)"

def check_package(name, min_version):
    """检查包是否安装"""
    try:
        module = importlib.import_module("cv2" if name == "cv2" else name)
        version = getattr(module, "__version__", "unknown")
        return True, f"{name} {version} ✓"
    except ImportError:
        return False, f"{name} ✗ (未安装)"

def check_environment():
    """检查所有环境依赖"""
    results = {"python": check_python_version()}
    for pkg, min_ver in REQUIRED_PACKAGES.items():
        results[pkg] = check_package(pkg, min_ver)
    return results

def print_report(results):
    """打印检查报告"""
    print("\n" + "="*60)
    print("环境检查报告")
    print("="*60)
    all_ok = all(ok for ok, _ in results.values())
    for name, (ok, msg) in results.items():
        print(f"  {msg}")
    print("="*60)
    print("✓ 环境检查通过" if all_ok else "✗ 环境存在问题")
    return all_ok

if __name__ == "__main__":
    results = check_environment()
    success = print_report(results)
    sys.exit(0 if success else 1)

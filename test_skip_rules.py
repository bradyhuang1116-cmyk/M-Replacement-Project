"""文件名首字母 → 跳框规则的端到端验证。

行为：对 TIF_Undo 下所有 TIF 跑 process_single_file，捕获日志中
  - "跳框规则 (首字母=X): 抑制 ..."
  - "检测到 N 个区域: [...]"
两条信息，最后打印对照表，校验是否符合：
  Y          → 无抑制
  B          → material_code_column
  P/G/O/J    → bottom_right_number, top_left_number
  其它       → 三者全抑制
"""
import os
import re
import sys
import logging
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from modules.docker_manager import ensure_vlm_ready
from modules.batch_processor import process_single_file
from modules.factory_note_pixel import clear_y_box_records

INPUT_DIR = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "skip_rules_smoke")
os.makedirs(OUT_DIR, exist_ok=True)


def expected_suppress(basename: str) -> set[str]:
    fl = basename[0].upper() if basename else ''
    if fl == 'Y':
        return set()
    if fl == 'B':
        return {"material_code_column"}
    if fl in ('P', 'G', 'O', 'J'):
        return {"bottom_right_number", "top_left_number"}
    return {"material_code_column", "bottom_right_number", "top_left_number"}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 捕获 batch_processor logger 的 INFO 日志到内存 buffer
    bp_logger = logging.getLogger("modules.batch_processor")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    bp_logger.addHandler(handler)

    ok, msg = ensure_vlm_ready()
    print(f"VLM ready={ok}: {msg}")
    if not ok:
        sys.exit(1)

    files = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if os.path.splitext(f)[1].lower() in (".tif", ".tiff")
    ]
    assert files, f"no TIFs in {INPUT_DIR}"
    print(f"待跑文件: {len(files)} 个")

    clear_y_box_records()

    summary = []
    for i, fp in enumerate(files, 1):
        base = os.path.basename(fp)
        print("=" * 80)
        print(f"[{i}/{len(files)}] {base}")
        buf_pos_before = buf.tell()
        try:
            process_single_file(fp, OUT_DIR)
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append((base, "EXC", None, None))
            continue
        buf.seek(buf_pos_before)
        text = buf.read()

        suppress_actual = set(re.findall(
            r"跳框规则 \(首字母=([^)]+)\): 抑制 (\S+)", text))
        suppress_keys = {key for (_, key) in suppress_actual}
        first_letter_seen = next((fl for (fl, _) in suppress_actual), None)

        # 末次 "检测到 N 个区域: [...]"
        m = re.findall(r"检测到 \d+ 个区域: \[([^\]]*)\]", text)
        kept_list = []
        if m:
            kept_list = [k.strip().strip("'\"") for k in m[-1].split(",") if k.strip()]
        kept_keys = set(kept_list)

        # 期望
        basename_noext = os.path.splitext(base)[0]
        exp_suppress = expected_suppress(basename_noext)
        exp_fl = basename_noext[0].upper() if basename_noext else '?'
        ok_match = suppress_keys == exp_suppress

        summary.append((base, exp_fl, suppress_keys, kept_keys, exp_suppress, ok_match))
        print(f"  首字母={exp_fl}  期望抑制={sorted(exp_suppress)}  实际抑制={sorted(suppress_keys)}  剩余={sorted(kept_keys - {'_metadata'})}  → {'OK' if ok_match else 'FAIL'}")

    print("=" * 80)
    print("汇总：")
    all_pass = True
    for row in summary:
        if len(row) == 4:
            base, status, _, _ = row
            print(f"  [{status}] {base}")
            all_pass = False
            continue
        base, fl, actual, kept, exp, ok = row
        mark = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{mark}] {base}  首字母={fl}  抑制={sorted(actual)}  剩余={sorted(kept - {'_metadata'})}")
    print(f"\n{'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()

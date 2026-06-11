"""Phase 2 烟雾测试 — JobQueue + Worker。

不依赖 Docker / VLM / PaddleOCR：
- 通过 monkey-patch 把 process_single_file 替换成 stub
- 用临时 sqlite + 临时输出目录
- 覆盖 enqueue → claim → mark_done / mark_failed / requeue 路径

跑法（项目根目录）：
    "C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_queue_worker_smoke.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke")


def _make_dummy_input(tmp_dir: Path) -> Path:
    """生成一个可被 process_single_file stub 接受的最小 TIF 占位文件。"""
    f = tmp_dir / "HYA000000-1_dummy-R.tif"
    f.write_bytes(b"not-a-real-tif-but-stub-doesnt-care")
    return f


def _stub_process_single_file_factory(should_fail_ids: set[int]):
    """返回一个会模拟 batch_processor.process_single_file 的桩函数。"""
    call_counter = {"n": 0}

    def stub(file_path: str, output_dir: str, **kwargs) -> dict:
        call_counter["n"] += 1
        call_id = call_counter["n"]
        log.info(f"  stub call #{call_id}: file={os.path.basename(file_path)} output={output_dir}")
        if call_id in should_fail_ids:
            raise RuntimeError(f"intentional failure on call #{call_id}")

        # 产物落到 output_dir/ocr/，模仿真实流水线
        ocr_dir = os.path.join(output_dir, "ocr")
        os.makedirs(ocr_dir, exist_ok=True)
        out_name = "H" + os.path.splitext(os.path.basename(file_path))[0] + "-R.tif"
        out_path = os.path.join(ocr_dir, out_name)
        Path(out_path).write_bytes(b"stub-output")
        return {
            "file": file_path,
            "status": "success",
            "method": "ocr",
            "replacements": [("YA000000", "HA000000")],
            "total": 1,
            "output_path": out_path,
            "regions_detected": ["material_code_column"],
        }

    return stub, call_counter


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="omc_queue_smoke_"))
    log.info(f"workdir={workdir}")
    db_path = workdir / "queue.db"
    output_root = workdir / "processed"
    input_dir = workdir / "input"
    input_dir.mkdir()

    # 必须在 import worker 之前，确保配置被识别（worker 持的是默认值，但
    # 这里不依赖默认值，我们通过构造参数传 output_root）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from modules import batch_processor
    from modules.job_queue import JobQueue
    from modules.worker import Worker

    # ── monkey-patch 处理函数 ──
    stub, counter = _stub_process_single_file_factory(should_fail_ids={2})
    batch_processor.process_single_file = stub  # type: ignore[assignment]

    # ── 准备输入文件 & 队列 ──
    f1 = _make_dummy_input(input_dir)
    f2 = _make_dummy_input(input_dir.joinpath("dummy2.tif").parent)
    f2 = input_dir / "HYB000000-1_dummy-R.tif"
    f2.write_bytes(b"x")
    f3 = input_dir / "HYC000000-1_dummy-R.tif"
    f3.write_bytes(b"x")

    queue = JobQueue(db_path=str(db_path))
    j1 = queue.enqueue("api", "manual", str(f1), drawing_no="YA000000", revision="1")
    j2 = queue.enqueue("watch_folder", "watch", str(f2), drawing_no="YB000000")
    j3 = queue.enqueue("api", "manual", str(f3))
    log.info(f"已入队 job ids: {[j1, j2, j3]}")
    assert queue.counts().get("pending") == 3, queue.counts()

    # ── 启动 Worker（关闭 VLM 预热，stub 不需要）──
    worker = Worker(
        queue,
        output_root=str(output_root),
        poll_interval=0.2,
        max_retry=1,
        ensure_vlm=False,
    )
    worker.start()

    # 等待全部任务被处理；j2 会失败一次后被 requeue，预计总共 4 次 stub 调用
    deadline = time.time() + 20
    while time.time() < deadline:
        c = queue.counts()
        if c.get("pending", 0) == 0 and c.get("running", 0) == 0:
            break
        time.sleep(0.2)
    worker.stop()

    counts = queue.counts()
    log.info(f"最终 counts={counts}; stub 调用次数={counter['n']}")

    # ── 断言 ──
    ok = True

    def check(cond: bool, msg: str):
        nonlocal ok
        status = "OK " if cond else "FAIL"
        log.info(f"  [{status}] {msg}")
        if not cond:
            ok = False

    check(counts.get("done", 0) == 3, f"done == 3 (got {counts.get('done', 0)})")
    check(counts.get("failed", 0) == 0, f"failed == 0 (got {counts.get('failed', 0)})")
    check(counter["n"] >= 4, f"stub 调用 ≥ 4（j2 失败→重试）实际={counter['n']}")

    jobs_done = queue.list_by_status("done")
    for j in jobs_done:
        check(bool(j.result_path) and os.path.exists(j.result_path),
              f"job {j.id} result_path 存在: {j.result_path}")
        check(j.error_msg is None, f"job {j.id} done 后 error_msg 应为 None")

    # j2 重试后 retry_count 应该 >= 1
    j2_obj = queue.get(j2)
    check(j2_obj is not None and j2_obj.retry_count >= 1,
          f"j2.retry_count ≥ 1 (got {j2_obj.retry_count if j2_obj else 'None'})")

    # ── 清理 ──
    try:
        shutil.rmtree(workdir)
    except OSError as e:
        log.warning(f"清理 tmp 目录失败（不影响结论）: {e}")

    log.info("=" * 60)
    log.info("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

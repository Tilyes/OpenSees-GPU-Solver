#!/usr/bin/env python3
"""summarize.py — 解析 results/*.log 中的 RESULT 行，生成对比报告 results/benchmark_report.md"""
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

LABELS = {
    "serial_superlu": "串行 OpenSees / SuperLU",
    "cuda_cg":        "CUDA / cusp CG+Jacobi",
    "sp_mumps_np4":   "OpenSeesSP / Mumps np=4",
    "sp_mumps_np8":   "OpenSeesSP / Mumps np=8",
}
SCENARIOS = {
    "s1": "场景1：3D 实体悬臂块·静力（18×18×200）",
    "s2": "场景2：3D 多层钢框架·静力（16×16×24）",
    "s3": "场景3：3D 实体块·瞬态 200 步（10×10×40）",
}


def parse_logs():
    data = {}  # (scenario, config) -> dict
    for log in sorted(RESULTS.glob("*.log")):
        m = re.match(r"(s\d)_(serial_superlu|cuda_cg|sp_mumps_np\d)", log.stem)
        if not m:
            continue
        scen, cfg = m.groups()
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith("RESULT"):
                continue
            if "DNF" in line or "FAILED" in line:
                data[(scen, cfg)] = {"dnf": True, "raw": line.strip()}
                continue
            kv = dict(re.findall(r"(\w+)=([^\s]+)", line))
            rec = {}
            for k in ("ndof", "np"):
                if k in kv:
                    rec[k] = int(float(kv[k]))
            for k in ("t_model", "t_analyze", "t_total", "t_gravity",
                      "t_transient", "t_perstep", "tipDisp"):
                if k in kv:
                    rec[k] = float(kv[k])
            if "nsteps" in kv:
                rec["nsteps"] = int(float(kv["nsteps"]))
            data[(scen, cfg)] = rec
    return data


def fmt_time(t):
    return f"{t:.2f}s" if t < 60 else f"{t/60:.2f}min"


def main():
    data = parse_logs()
    if not data:
        print("no RESULT lines found", file=sys.stderr)
        sys.exit(1)

    lines = ["# OpenSees 三版本性能对比报告", ""]
    for scen, title in SCENARIOS.items():
        base = data.get((scen, "serial_superlu"), {})
        base_disp = base.get("tipDisp")
        # 主时间指标：s1/s2 用 t_analyze，s3 用 t_transient
        key = "t_transient" if scen == "s3" else "t_analyze"
        t_base = base.get(key)

        lines += [f"## {title}", ""]
        ndof = next((r["ndof"] for (s, _), r in data.items()
                     if s == scen and "ndof" in r), None)
        if ndof:
            lines.append(f"DOF 总数：**{ndof:,}**")
            lines.append("")
        lines.append("| 配置 | 时间 | 加速比 | tip 位移 | 数值偏差 |")
        lines.append("|---|---|---|---|---|")
        for cfg, label in LABELS.items():
            r = data.get((scen, cfg))
            if r is None:
                lines.append(f"| {label} | — | — | — | — |")
                continue
            if r.get("dnf"):
                lines.append(f"| {label} | **DNF** | — | — | — |")
                continue
            t = r.get(key)
            disp = r.get("tipDisp")
            speedup = f"{t_base/t:.2f}×" if (t_base and t) else "—"
            dev = "—"
            if base_disp is not None and disp is not None and base_disp != 0:
                dev = f"{abs(disp-base_disp)/abs(base_disp):.2e}"
            disp_s = f"{disp:.10g}" if disp is not None else "—"
            lines.append(f"| {label} | {fmt_time(t)} | {speedup} | {disp_s} | {dev} |")
        lines.append("")

    out = RESULTS / "benchmark_report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()

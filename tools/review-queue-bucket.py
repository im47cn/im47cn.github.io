#!/usr/bin/env python3
"""review-queue-bucket.py — 双审排队量化分桶（手册 v2.2 · Deploy 章节 Q5 前置）

回答两个问题（对应 Q5 falsifier）：
  1. 核心链路 PR 占比是多少？（git 模式即可，纯本地，无外部 API）
  2. 核心链路的 review 排队时长是否显著高于非核心？（需 Codeup 侧的 PR 时间戳，走 csv 模式）

用法：
  # 模式一：git 占比 + review 尾延迟代理（merge 时刻 − feature 分支末次提交时刻）
  python3 review-queue-bucket.py git --repo ~/repo/order-service --base develop \
      --since 2026-07-12 --until 2026-10-12 \
      --core-paths "src/main/java/com/x/order,src/main/java/com/x/pay"

  # 模式二：csv 精确排队时长（Codeup 导出或 OpenAPI 落成 CSV）
  #   必需列：mr_id,path_class,opened_at,first_review_at,merged_at
  #   path_class 取值：core / noncore；时间用 ISO 格式或 epoch 秒
  python3 review-queue-bucket.py csv --file prs.csv --since 2026-07-12 --until 2026-10-12

  # 自检（ponytail 式：单文件单断言组，非测试套件）
  python3 review-queue-bucket.py selftest

  # 工件校验（E2E 最低标准：golden 字节比对，重跑两次结果一致）
  python3 review-queue-bucket.py git ... --golden expected.txt   # 不一致 exit 2

注意：
  - 时间窗必须显式给出（禁相对 now），保证重跑结果一致。
  - Codeup OpenAPI 的端点/字段名未经本脚本验证——拉取权威文档后自行落成 CSV 再喂 csv 模式，
    不要对着未知 schema 盲试端点（全局门禁：同端点连续 2 次失败即停）。
  - review 尾延迟代理的含义：merge − 分支末次提交 ≈ 最后一轮 review 的翻转时长，
    不含 PR 打开到首次 review 的完整排队。精确排队必须走 csv 模式。

输出：确定性文本报表。同输入同输出，无随机、无时间戳。
"""

import argparse
import csv
import datetime
import subprocess
import sys

ISO_FMTS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


def classify(changed_files, core_prefixes):
    """任一文件落在核心前缀下 → core，否则 noncore。前缀 "." 或 "" 表示整仓核心/整仓非核心。"""
    if "." in core_prefixes or "" in core_prefixes:
        return "core" if "." in core_prefixes else "noncore"
    for f in changed_files:
        for p in core_prefixes:
            if f == p or f.startswith(p.rstrip("/") + "/"):
                return "core"
    return "noncore"


def pct(sorted_vals, p):
    """确定性百分位：线性插值，p∈[0,100]。空列表返回 None。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def parse_ts(s):
    s = s.strip()
    if s.isdigit():
        return int(s)
    for f in ISO_FMTS:
        try:
            return int(datetime.datetime.strptime(s.replace("Z", "+0000"), f).timestamp())
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {s}")


def run_git(repo, *args):
    out = subprocess.run(["git", "-C", repo] + list(args),
                         capture_output=True, text=True, check=True)
    return out.stdout


def git_mode(repo, base, since, until, core_prefixes):
    fmt = "%H%x00%ct%x00%s"
    lines = run_git(repo, "log", "--merges", f"--since={since}", f"--until={until}",
                    "--date-order", f"--format={fmt}", base).splitlines()
    rows = []  # (mr_label, class, merge_ct, tail_latency_seconds, files)
    for line in lines:
        parts = line.split("\x00")
        if len(parts) < 3:
            continue
        sha, ct, subject = parts[0], int(parts[1]), parts[2]
        files = [f for f in run_git(repo, "diff", "--name-only", f"{sha}^1", sha).splitlines() if f]
        try:
            tail = run_git(repo, "log", "-1", "--format=%ct", f"{sha}^2").strip()
            tail_latency = max(0, ct - int(tail))
        except subprocess.CalledProcessError:
            tail_latency = None  # octopus 等无第二父场景，跳过代理值
        rows.append((subject, classify(files, core_prefixes), ct, tail_latency, len(files)))
    return rows


def render_git_report(rows, since, until, base):
    out = []
    out.append(f"== 双审量化分桶（git 模式）base={base} 窗口=[{since} ~ {until}] ==")
    total = len(rows)
    if total == 0:
        out.append("窗口内无 merge 记录，占比与分桶不可判定（falsifier 未触发，数据不足）。")
        return "\n".join(out)
    core_rows = [r for r in rows if r[1] == "core"]
    share = len(core_rows) / total * 100
    out.append(f"PR 总数: {total}")
    out.append(f"核心链路 PR: {len(core_rows)} ({share:.1f}%)  |  非核心: {total - len(core_rows)} ({100 - share:.1f}%)")
    out.append(f"falsifier 判据: 核心占比 <20% 且排队无显著差异 => 分层是过度工程，本题作废")
    out.append("")
    out.append(f"{'桶':<10}{'数量':>6}{'占比':>8}{'尾延迟中位(min)':>18}{'尾延迟P90(min)':>16}{'diff文件数中位':>14}")
    for label, bucket in (("core", core_rows), ("noncore", [r for r in rows if r[1] == "noncore"])):
        if not bucket:
            out.append(f"{label:<10}{0:>6}{0.0:>7.1f}%{'-':>18}{'-':>16}{'-':>14}")
            continue
        tails = sorted(r[3] / 60.0 for r in bucket if r[3] is not None)
        sizes = sorted(float(r[4]) for r in bucket)
        m = f"{pct(tails, 50):.0f}" if tails else "-"
        p90 = f"{pct(tails, 90):.0f}" if tails else "-"
        med_size = f"{pct(sizes, 50):.0f}"
        out.append(f"{label:<10}{len(bucket):>6}{len(bucket) / total * 100:>7.1f}%{m:>18}{p90:>16}{med_size:>14}")
    return "\n".join(out)


def csv_mode(path, since, until):
    rows = []  # (class, queue_minutes, cycle_minutes)
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            opened = parse_ts(rec["opened_at"])
            if not (parse_ts(since) <= opened <= parse_ts(until)):
                continue
            queue = parse_ts(rec["first_review_at"]) - opened
            cycle = parse_ts(rec["merged_at"]) - opened
            rows.append((rec["path_class"].strip(), max(0, queue / 3600.0), max(0, cycle / 3600.0)))
    return rows


def render_csv_report(rows, since, until):
    out = []
    out.append(f"== 双审量化分桶（csv 模式）窗口=[{since} ~ {until}] ==")
    total = len(rows)
    if total == 0:
        out.append("窗口内无 PR 记录，不可判定。")
        return "\n".join(out)
    out.append(f"PR 总数: {total}")
    out.append("")
    out.append(f"{'桶':<10}{'数量':>6}{'占比':>8}{'排队中位(h)':>13}{'排队P90(h)':>12}{'周期中位(h)':>13}")
    for label in ("core", "noncore"):
        bucket = [r for r in rows if r[0] == label]
        if not bucket:
            out.append(f"{label:<10}{0:>6}{0.0:>7.1f}%{'-':>13}{'-':>12}{'-':>13}")
            continue
        queues = sorted(r[1] for r in bucket)
        cycles = sorted(r[2] for r in bucket)
        out.append(f"{label:<10}{len(bucket):>6}{len(bucket) / total * 100:>7.1f}%"
                   f"{pct(queues, 50):>13.1f}{pct(queues, 90):>12.1f}{pct(cycles, 50):>13.1f}")
    out.append("")
    out.append("判读: core 排队中位显著高于 noncore（>1.5x 且绝对值 >4h）=> Q5 分档启用；否则分层作废。")
    return "\n".join(out)


def selftest():
    # classify
    assert classify(["src/legacy/a.java"], ["src/order"]) == "noncore"
    assert classify(["src/order/Op.java", "docs/x.md"], ["src/order"]) == "core"
    assert classify(["src/orderish/a.java"], ["src/order"]) == "noncore"  # 前缀不得含糊匹配
    assert classify(["any/file.java"], ["."]) == "core"    # 整仓核心（如网关仓）
    assert classify(["any/file.java"], [""]) == "noncore"  # 整仓非核心（如后台仓）
    # pct
    assert pct([], 50) is None
    assert pct([3.0], 50) == 3.0
    assert pct([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert pct([1.0, 2.0, 3.0, 4.0], 90) == 3.7
    # parse_ts
    assert parse_ts("1700000000") == 1700000000
    assert parse_ts("2026-10-12") == int(datetime.datetime(2026, 10, 12).timestamp())
    # render 空窗口确定性
    r1 = render_git_report([], "2026-07-12", "2026-10-12", "develop")
    r2 = render_git_report([], "2026-07-12", "2026-10-12", "develop")
    assert r1 == r2 and "不可判定" in r1
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("git", help="纯本地 git：核心占比 + review 尾延迟代理")
    g.add_argument("--repo", required=True)
    g.add_argument("--base", default="develop")
    g.add_argument("--since", required=True, help="YYYY-MM-DD，显式窗口起点")
    g.add_argument("--until", required=True, help="YYYY-MM-DD，显式窗口终点")
    g.add_argument("--core-paths", required=True, help="逗号分隔的核心路径前缀")
    g.add_argument("--golden", help="golden 文件字节比对，不一致 exit 2")

    c = sub.add_parser("csv", help="Codeup 导出 CSV：精确排队时长")
    c.add_argument("--file", required=True)
    c.add_argument("--since", required=True)
    c.add_argument("--until", required=True)
    c.add_argument("--golden")

    sub.add_parser("selftest", help="ponytail 自检")
    args = ap.parse_args()

    if args.mode == "selftest":
        selftest()
        return
    if args.mode == "git":
        prefixes = [p.strip() for p in args.core_paths.split(",") if p.strip()]
        report = render_git_report(git_mode(args.repo, args.base, args.since, args.until, prefixes),
                                   args.since, args.until, args.base)
    else:
        report = render_csv_report(csv_mode(args.file, args.since, args.until), args.since, args.until)

    print(report)
    if getattr(args, "golden", None):
        with open(args.golden, encoding="utf-8") as fh:
            # print() 会在报告末尾追加一个换行，golden 容许恰好一个尾换行
            if fh.read().rstrip("\n") != report:
                print("GOLDEN MISMATCH", file=sys.stderr)
                sys.exit(2)
        print("golden 比对一致")


if __name__ == "__main__":
    main()

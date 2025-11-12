#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import subprocess

DOC_TL_CN = "\u8bad\u7ec3\u65f6\u95f4\u7ebf_20251111.md"
DOC_REPORT = "\u5b9e\u9a8c\u62a5\u544a.txt"

def tail_last_line(path: str) -> str:
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(4096, size)
            f.seek(size - read_size)
            data = f.read().decode('utf-8', 'ignore')
        lines = [l for l in data.splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""

def has_finished_100(line: str) -> bool:
    return ("Epoch 100" in line) or ("Epoch 099" in line and "test_acc" in line)

def visualize(task: str, log_path: str):
    cmd = [
        "python3", "-u", "src/visualize.py",
        "--task", task,
        "--log_path", log_path,
    ]
    subprocess.run(cmd, check=True)

def append_to_docs(task: str, log_path: str, curves_hint: str, cm_hint: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = (
        f"\n- auto-append: {ts} - `{log_path}`\n"
        f"  - task: {task}\n"
        f"  - figures: `{curves_hint}`, `{cm_hint}`\n"
    )
    with open(DOC_TL_CN, 'a', encoding='utf-8') as f:
        f.write(entry)
    with open(DOC_REPORT, 'a', encoding='utf-8') as f:
        f.write("\nAuto-append: Chinese 100-epoch run figures generated.\n" + entry)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['chinese'])
    parser.add_argument('--logs', nargs='+', required=True)
    args = parser.parse_args()

    logs_done = set()
    print("[watch] start:", args.logs)
    while True:
        for lp in args.logs:
            if lp in logs_done:
                continue
            last = tail_last_line(lp)
            if has_finished_100(last):
                print(f"[watch] detected completion for {lp}. Generating visuals...")
                visualize(args.task, lp)
                vis_dir = os.path.join('outputs', 'visuals')
                curves = sorted(
                    [os.path.join('outputs', 'visuals', f) for f in os.listdir(vis_dir) if f.startswith('curves_')],
                    key=os.path.getmtime
                )
                cms = sorted(
                    [os.path.join('outputs', 'visuals', f) for f in os.listdir(vis_dir) if f.startswith('cm_')],
                    key=os.path.getmtime
                )
                curves_hint = curves[-1] if curves else 'outputs/visuals/curves_latest.png'
                cm_hint = cms[-1] if cms else 'outputs/visuals/cm_latest.png'
                append_to_docs(args.task, lp, curves_hint, cm_hint)
                logs_done.add(lp)
        if len(logs_done) == len(args.logs):
            print("[watch] all logs processed. exiting.")
            break
        time.sleep(5)

if __name__ == '__main__':
    main()
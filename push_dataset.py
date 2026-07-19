#!/usr/bin/env python3
"""Resumable push of a local LeRobotDataset to the HF Hub.

Uses upload_large_folder (multi-threaded, resumable, retries on flaky networks).
Safe to re-run — it skips already-uploaded files and continues.

    /home/shiv/miniforge3/envs/lerobot/bin/python push_dataset.py \
        --repo-id Shivakumr/yams [--private]
"""
import argparse
import os

from huggingface_hub import HfApi, upload_large_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--local", default=None)
    args = ap.parse_args()

    local = args.local or os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    if not os.path.isdir(local):
        raise SystemExit(f"local dataset not found: {local}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    print(f"uploading {local} -> {args.repo_id} (resumable)...", flush=True)
    upload_large_folder(folder_path=local, repo_id=args.repo_id, repo_type="dataset")
    print(f"DONE: https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()

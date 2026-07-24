# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pre-commit check to ensure protobuf files (.proto) and generated code (_pb2.py) remain in sync."""

import os
import subprocess
import sys


def check_proto_sync() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    proto_file = os.path.join(repo_root, "vllm", "entrypoints", "pull_worker", "proto", "queue.proto")

    if not os.path.exists(proto_file):
        return 0

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{repo_root}",
        f"--python_out={repo_root}",
        f"--grpc_python_out={repo_root}",
        proto_file,
    ]

    try:
        subprocess.run(cmd, check=True, cwd=repo_root)
    except Exception as e:
        print(f"Error executing protoc compilation: {e}", file=sys.stderr)
        return 1

    status_cmd = ["git", "status", "--porcelain", "vllm/entrypoints/pull_worker/proto/"]
    result = subprocess.run(status_cmd, capture_output=True, text=True, cwd=repo_root)

    if result.stdout.strip():
        print("ERROR: Generated Protobuf code is out of sync with .proto specification!", file=sys.stderr)
        print("Modified files detected after protoc compilation:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("Please re-run compilation: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vllm/entrypoints/pull_worker/proto/queue.proto", file=sys.stderr)
        return 1

    print("Protobuf files and generated code are 100% in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(check_proto_sync())

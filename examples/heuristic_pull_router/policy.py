# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Custom Prefix-Cache Locality & SLA Priority Aware Routing Policy."""

import time
import zlib
from typing import Any, Dict, List, Set

from examples.heuristic_pull_router.framework import BasePolicy
from vllm.entrypoints.pull_worker.proto import queue_pb2


def compute_prompt_prefix_hash(prompt: str, prefix_len: int = 64) -> int:
    """Compute a simple 64-bit CRC hash representation for the prompt prefix."""
    prefix = prompt[:prefix_len].encode("utf-8")
    return zlib.crc32(prefix) & 0xFFFFFFFF


class PrefixAndSLAAwarePolicy(BasePolicy):
    """Custom pluggable routing policy for vLLM Disaggregated Pull Queue.

    Scores and prioritizes pending requests based on:
    1. Prefix Cache Locality: Matching request prompt prefix hashes with the worker's
       active radix cache block hashes to maximize KV cache reuse.
    2. Multi-Tenant SLA Priority: Prioritizing VIP/High-Priority tenant requests.
    3. Anti-Starvation Wait Time: Boosting older pending tasks to guarantee SLA bounds.
    """

    def __init__(
        self,
        prefix_weight: float = 1000.0,
        priority_weight: float = 500.0,
        age_weight: float = 10.0,
    ):
        self.prefix_weight = prefix_weight
        self.priority_weight = priority_weight
        self.age_weight = age_weight

    def rank_tasks(
        self,
        worker_status: queue_pb2.WorkerStatus,
        pending_tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pure ranking function called by the framework when a worker requests work.

        Args:
            worker_status: The latest status reported by the pulling vLLM worker.
            pending_tasks: Snapshot list of all unassigned pending task dictionaries.

        Returns:
            Sorted list of task dictionaries ordered by descending score.
        """
        # Extract worker's active prefix block hashes from extended metrics
        active_hashes: Set[int] = set()
        ext = worker_status.extended_metrics
        if ext and ext.HasField("prefix_cache_stats"):
            active_hashes = set(ext.prefix_cache_stats.active_prefix_hashes)

        now = time.time()
        scored_tasks = []

        for task in pending_tasks:
            # 1. Calculate Prefix Locality Score
            prompt = task.get("prompt", "")
            prompt_hash = compute_prompt_prefix_hash(prompt)
            prefix_score = (
                self.prefix_weight if prompt_hash in active_hashes else 0.0
            )

            # 2. Calculate SLA Priority Score (e.g., VIP priority=10, Standard priority=0)
            priority = float(task.get("priority", 0))
            priority_score = priority * self.priority_weight

            # 3. Calculate Anti-Starvation Wait Time Score
            created_at = task.get("created_at", now)
            wait_seconds = max(0.0, now - created_at)
            age_score = wait_seconds * self.age_weight

            total_score = prefix_score + priority_score + age_score

            scored_tasks.append((total_score, task))

        # Sort tasks in descending order of total score
        scored_tasks.sort(key=lambda item: item[0], reverse=True)
        return [task for _, task in scored_tasks]

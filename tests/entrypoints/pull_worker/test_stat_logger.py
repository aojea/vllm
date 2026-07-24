# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for GrpcQueueWorkerStatLogger."""

from unittest.mock import MagicMock

from vllm.entrypoints.pull_worker.proto import queue_pb2
from vllm.entrypoints.pull_worker.stat_logger import GrpcQueueWorkerStatLogger
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats


def test_stat_logger_standard_metrics():
    """Verify standard metrics recording from SchedulerStats."""
    mock_config = MagicMock()
    logger_inst = GrpcQueueWorkerStatLogger(vllm_config=mock_config)

    stats = SchedulerStats(
        num_running_reqs=5,
        num_waiting_reqs=12,
        kv_cache_usage=0.45,
    )

    logger_inst.record(scheduler_stats=stats, iteration_stats=None)
    worker_status = logger_inst.get_latest_worker_status(worker_id="test-worker-1")

    assert worker_status.worker_id == "test-worker-1"
    assert worker_status.standard_metrics.num_running_reqs == 5
    assert worker_status.standard_metrics.num_waiting_reqs == 12
    assert abs(worker_status.standard_metrics.gpu_kv_cache_usage_pct - 0.45) < 1e-5


def test_stat_logger_extended_metrics_opt_in():
    """Verify extended metrics and active prefix hashes when opted in."""
    mock_config = MagicMock()
    logger_inst = GrpcQueueWorkerStatLogger(
        vllm_config=mock_config,
        extended_metrics_config={"prefix_cache", "perf_latency"},
    )

    prefix_stats = PrefixCacheStats(
        requests=10,
        queries=100,
        hits=80,
    )
    # Set active prefix block hashes for prefix-aware routing
    prefix_stats.active_prefix_hashes = [123456789, 987654321]

    stats = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=0,
        kv_cache_usage=0.1,
        prefix_cache_stats=prefix_stats,
    )

    logger_inst.record(scheduler_stats=stats, iteration_stats=None)
    worker_status = logger_inst.get_latest_worker_status(worker_id="test-worker-2")

    ext = worker_status.extended_metrics
    assert ext.prefix_cache_stats.queries == 100
    assert ext.prefix_cache_stats.hits == 80
    assert list(ext.prefix_cache_stats.active_prefix_hashes) == [123456789, 987654321]


def test_stat_logger_custom_escape_hatch():
    """Verify custom metrics and labels escape hatch."""
    mock_config = MagicMock()
    logger_inst = GrpcQueueWorkerStatLogger(vllm_config=mock_config)

    logger_inst.set_custom_metric("gpu_power_watts", 250.5)
    logger_inst.set_custom_label("gpu_model", "H100")

    stats = SchedulerStats(num_running_reqs=1, num_waiting_reqs=0, kv_cache_usage=0.05)
    logger_inst.record(scheduler_stats=stats, iteration_stats=None)
    worker_status = logger_inst.get_latest_worker_status(worker_id="test-worker-3")

    ext = worker_status.extended_metrics
    assert ext.custom_metrics["gpu_power_watts"] == 250.5
    assert ext.custom_labels["gpu_model"] == "H100"

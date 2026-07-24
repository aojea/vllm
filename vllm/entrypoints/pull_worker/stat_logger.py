# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""gRPC Queue Worker Stat Logger for vLLM."""

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm.entrypoints.pull_worker.proto import queue_pb2
from vllm.logger import init_logger
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

logger = init_logger(__name__)


class GrpcQueueWorkerStatLogger(StatLoggerBase):
    """Stat logger that collects engine metrics and constructs WorkerStatus

    protobuf messages to report to the central pull queue load balancer.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_index: int = 0,
        extended_metrics_config: set[str] | None = None,
    ):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.extended_metrics_config = extended_metrics_config or set()

        self._latest_status: queue_pb2.WorkerStatus | None = None
        self._custom_metrics: dict[str, float] = {}
        self._custom_labels: dict[str, str] = {}

    def log_engine_initialized(self) -> None:
        logger.info(
            "GrpcQueueWorkerStatLogger initialized for engine index %d",
            self.engine_index,
        )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None:
            return

        # 1. Populate Standard Metrics (Non-negotiable for load balancing)
        std_metrics = queue_pb2.StandardMetrics(
            num_running_reqs=scheduler_stats.num_running_reqs,
            num_waiting_reqs=scheduler_stats.num_waiting_reqs,
            free_gpu_kv_blocks=0,  # Computed if raw block count available
            gpu_kv_cache_usage_pct=scheduler_stats.kv_cache_usage,
            is_paused=False,
            is_sleeping=False,
        )

        # 2. Populate Extended Metrics based on opt-in configuration
        ext_metrics = queue_pb2.ExtendedMetrics()
        enable_all = "all" in self.extended_metrics_config

        if enable_all or "prefix_cache" in self.extended_metrics_config:
            prefix_stats = scheduler_stats.prefix_cache_stats
            hashes: list[int] = []
            if (
                hasattr(prefix_stats, "active_prefix_hashes")
                and prefix_stats.active_prefix_hashes
            ):
                hashes = list(prefix_stats.active_prefix_hashes)

            ext_metrics.prefix_cache_stats.CopyFrom(
                queue_pb2.PrefixCacheStats(
                    hit_rate=getattr(prefix_stats, "hit_rate", 0.0),
                    queries=getattr(prefix_stats, "queries", 0),
                    hits=getattr(prefix_stats, "hits", 0),
                    active_prefix_hashes=hashes,
                )
            )

        if (
            enable_all or "perf_latency" in self.extended_metrics_config
        ) and scheduler_stats.perf_stats:
            ext_metrics.perf_latency_stats.CopyFrom(
                queue_pb2.PerfLatencyStats(
                    avg_time_to_first_token_ms=getattr(
                        scheduler_stats.perf_stats, "avg_ttft_ms", 0.0
                    ),
                    avg_inter_token_latency_ms=getattr(
                        scheduler_stats.perf_stats, "avg_itl_ms", 0.0
                    ),
                    e2e_latency_ms=getattr(
                        scheduler_stats.perf_stats, "e2e_latency_ms", 0.0
                    ),
                )
            )

        if (
            enable_all or "kv_transfer" in self.extended_metrics_config
        ) and scheduler_stats.kv_connector_stats:
            ext_metrics.kv_connector_stats.CopyFrom(
                queue_pb2.KVConnectorStats(
                    transferred_bytes=scheduler_stats.kv_connector_stats.get(
                        "transferred_bytes", 0
                    ),
                    avg_transfer_latency_ms=scheduler_stats.kv_connector_stats.get(
                        "avg_latency_ms", 0.0
                    ),
                )
            )

        if (
            enable_all or "spec_decoding" in self.extended_metrics_config
        ) and scheduler_stats.spec_decoding_stats:
            ext_metrics.spec_decoding_stats.CopyFrom(
                queue_pb2.SpecDecodingStats(
                    draft_acceptance_rate=getattr(
                        scheduler_stats.spec_decoding_stats, "acceptance_rate", 0.0
                    ),
                    draft_tokens_generated=getattr(
                        scheduler_stats.spec_decoding_stats, "num_draft_tokens", 0
                    ),
                    draft_tokens_accepted=getattr(
                        scheduler_stats.spec_decoding_stats,
                        "num_accepted_tokens",
                        0,
                    ),
                )
            )

        # 3. Populate Extensibility Escape Hatch
        for k, v in self._custom_metrics.items():
            ext_metrics.custom_metrics[k] = float(v)
        for k, v in self._custom_labels.items():
            ext_metrics.custom_labels[k] = str(v)

        # Construct WorkerStatus
        self._latest_status = queue_pb2.WorkerStatus(
            worker_id="",  # Set dynamically by worker
            timestamp_unix_ms=int(time.time() * 1000),
            standard_metrics=std_metrics,
            extended_metrics=ext_metrics,
        )

    def set_custom_metric(self, name: str, value: float) -> None:
        self._custom_metrics[name] = value

    def set_custom_label(self, name: str, value: str) -> None:
        self._custom_labels[name] = value

    def get_latest_worker_status(self, worker_id: str) -> queue_pb2.WorkerStatus:
        if self._latest_status is None:
            return queue_pb2.WorkerStatus(
                worker_id=worker_id,
                timestamp_unix_ms=int(time.time() * 1000),
                standard_metrics=queue_pb2.StandardMetrics(),
            )
        status = queue_pb2.WorkerStatus()
        status.CopyFrom(self._latest_status)
        status.worker_id = worker_id
        return status

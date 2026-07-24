# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Production Framework for vLLM Disaggregated Pull Queue with Pluggable Policies."""

from abc import ABC, abstractmethod
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import grpc
from google.protobuf.empty_pb2 import Empty

# Import compiled vLLM pull_worker protobuf definitions
from vllm.entrypoints.pull_worker.proto import queue_pb2, queue_pb2_grpc

logger = logging.getLogger("pull_queue_framework")

# Optional Redis integration
try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    aioredis = None  # type: ignore[assignment]


class BasePolicy(ABC):
    """Abstract Base Class for pluggable pull-queue routing policies.

    Policy implementations must remain pure functions: taking the worker's
    latest capacity/metrics status and a snapshot of all pending queue tasks,
    and returning a prioritized list of tasks for that worker to execute.
    """

    @abstractmethod
    def rank_tasks(
        self,
        worker_status: queue_pb2.WorkerStatus,
        pending_tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter and rank pending tasks for a pulling worker.

        Args:
            worker_status: Latest WorkerStatus reported by the pulling worker.
            pending_tasks: List of raw pending task dictionaries.

        Returns:
            List of prioritized task dictionaries ordered by execution preference.
        """
        pass


class PullQueueFramework(queue_pb2_grpc.PullWorkerServiceServicer):
    """Production gRPC & Redis Framework for vLLM Disaggregated Pull Queue.

    Hides all distributed complexity (Redis connectivity, task serialization,
    state tracking, atomic popping, and streaming channels) behind a clean
    pluggable policy interface.
    """

    REDIS_KEY_PENDING = "vllm:pull_queue:pending"
    REDIS_KEY_WORKER_PREFIX = "vllm:pull_queue:worker:"

    def __init__(
        self,
        policy: BasePolicy,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        self.policy = policy
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_client: Optional[Any] = None

        # In-memory storage fallback if Redis is not configured or unavailable
        self._in_memory_pending_tasks: List[Dict[str, Any]] = []
        self._in_memory_workers: Dict[str, queue_pb2.WorkerStatus] = {}
        self._in_memory_lock = asyncio.Lock()

        # Output streams: request_id -> list of asyncio.Queue for streaming response chunks
        self._output_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._subscribers_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Redis connection or in-memory fallback storage."""
        if self.redis_host and HAS_REDIS:
            try:
                self.redis_client = aioredis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    decode_responses=True,
                )
                await self.redis_client.ping()
                logger.info(
                    "Connected to Redis state backend at %s:%d (db %d)",
                    self.redis_host,
                    self.redis_port,
                    self.redis_db,
                )
            except Exception as e:
                logger.warning(
                    "Failed to connect to Redis at %s:%d (%s). Falling back to in-memory storage.",
                    self.redis_host,
                    self.redis_port,
                    e,
                )
                self.redis_client = None
        else:
            logger.info("Using in-memory queue storage backend.")

    # -------------------------------------------------------------------------
    # Public Client Interface (Enqueue & Output Streaming)
    # -------------------------------------------------------------------------

    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a new task into the queue.

        Args:
            task_data: Task payload dictionary containing 'request_id', 'prompt',
                       'sampling_params', and optional 'priority' / metadata.

        Returns:
            The unique request_id assigned to the task.
        """
        if "request_id" not in task_data or not task_data["request_id"]:
            task_data["request_id"] = f"req-{uuid.uuid4().hex[:8]}"

        if "created_at" not in task_data:
            task_data["created_at"] = time.time()

        task_json = json.dumps(task_data)

        if self.redis_client:
            await self.redis_client.rpush(self.REDIS_KEY_PENDING, task_json)
        else:
            async with self._in_memory_lock:
                self._in_memory_pending_tasks.append(task_data)

        logger.info("Enqueued task %s into pull queue", task_data["request_id"])
        return task_data["request_id"]

    async def subscribe_outputs(self, request_id: str) -> AsyncIterator[queue_pb2.RequestOutputChunk]:
        """Subscribe to streaming output chunks for a specific request ID."""
        output_queue: asyncio.Queue = asyncio.Queue()
        async with self._subscribers_lock:
            if request_id not in self._output_subscribers:
                self._output_subscribers[request_id] = []
            self._output_subscribers[request_id].append(output_queue)

        try:
            while True:
                chunk: Optional[queue_pb2.RequestOutputChunk] = await output_queue.get()
                if chunk is None:  # End of stream sentinel
                    break
                yield chunk
        finally:
            async with self._subscribers_lock:
                if request_id in self._output_subscribers:
                    if output_queue in self._output_subscribers[request_id]:
                        self._output_subscribers[request_id].remove(output_queue)
                    if not self._output_subscribers[request_id]:
                        del self._output_subscribers[request_id]

    async def _broadcast_output(self, chunk: queue_pb2.RequestOutputChunk) -> None:
        """Broadcast output chunk to all active subscribers of the request."""
        async with self._subscribers_lock:
            subscribers = list(self._output_subscribers.get(chunk.request_id, []))

        for q in subscribers:
            await q.put(chunk)

    # -------------------------------------------------------------------------
    # gRPC Service Implementation (Worker RPC Interface)
    # -------------------------------------------------------------------------

    async def RegisterWorker(
        self,
        request: queue_pb2.RegisterWorkerRequest,
        context: grpc.aio.ServicerContext,
    ) -> queue_pb2.RegisterWorkerResponse:
        """Register a new vLLM Pull Worker instance."""
        meta = request.metadata
        logger.info(
            "Worker %s registered (Model: %s, MaxLen: %d, WorldSize: %d)",
            meta.worker_id,
            meta.model_name,
            meta.max_model_len,
            meta.world_size,
        )
        session_token = f"sess-{uuid.uuid4().hex[:12]}"
        return queue_pb2.RegisterWorkerResponse(
            accepted=True,
            session_token=session_token,
            assigned_worker_id=meta.worker_id,
        )

    async def ReportStatus(
        self,
        request: queue_pb2.WorkerStatus,
        context: grpc.aio.ServicerContext,
    ) -> queue_pb2.ReportStatusResponse:
        """Store status and metrics update from worker."""
        if self.redis_client:
            key = f"{self.REDIS_KEY_WORKER_PREFIX}{request.worker_id}"
            await self.redis_client.set(key, request.SerializeToString(), ex=30)
        else:
            async with self._in_memory_lock:
                self._in_memory_workers[request.worker_id] = request

        return queue_pb2.ReportStatusResponse(acknowledged=True)

    async def StreamTasks(
        self,
        request_iterator: AsyncIterator[queue_pb2.TaskGrantRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[queue_pb2.GenerateRequest]:
        """Bidirectional streaming RPC: Workers push grant credits, Queue yields ranked tasks."""
        async for grant in request_iterator:
            slots = grant.slots_available
            if slots <= 0:
                continue

            worker_id = grant.worker_id
            worker_status = await self._get_worker_status(worker_id)
            if worker_status is None:
                worker_status = queue_pb2.WorkerStatus(
                    worker_id=worker_id,
                    standard_metrics=queue_pb2.StandardMetrics(num_running_reqs=0),
                )

            # Fetch pending tasks and rank using policy
            pending_tasks = await self._get_all_pending_tasks()
            if not pending_tasks:
                continue

            ranked_tasks = self.policy.rank_tasks(worker_status, pending_tasks)
            if not ranked_tasks:
                continue

            # Pop top `slots` tasks and send down stream to worker
            for task_dict in ranked_tasks[:slots]:
                popped_task = await self._pop_task_by_id(task_dict["request_id"])
                if popped_task is None:
                    continue  # Task taken by another worker race condition

                proto_req = self._dict_to_generate_request(popped_task)
                logger.info(
                    "Dispatched task %s to pull worker %s",
                    proto_req.request_id,
                    worker_id,
                )
                yield proto_req

    async def StreamOutput(
        self,
        request_iterator: AsyncIterator[queue_pb2.RequestOutputChunk],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        """Stream token output chunks from worker to central queue server."""
        async for chunk in request_iterator:
            await self._broadcast_output(chunk)
            yield Empty()

    # -------------------------------------------------------------------------
    # Internal Task Storage Operations
    # -------------------------------------------------------------------------

    async def _get_worker_status(self, worker_id: str) -> Optional[queue_pb2.WorkerStatus]:
        if self.redis_client:
            key = f"{self.REDIS_KEY_WORKER_PREFIX}{worker_id}"
            data = await self.redis_client.get(key)
            if data:
                status = queue_pb2.WorkerStatus()
                status.ParseFromString(data)
                return status
            return None
        else:
            async with self._in_memory_lock:
                return self._in_memory_workers.get(worker_id)

    async def _get_all_pending_tasks(self) -> List[Dict[str, Any]]:
        if self.redis_client:
            raw_list = await self.redis_client.lrange(self.REDIS_KEY_PENDING, 0, -1)
            tasks = []
            for item in raw_list:
                try:
                    tasks.append(json.loads(item))
                except Exception:
                    pass
            return tasks
        else:
            async with self._in_memory_lock:
                return list(self._in_memory_pending_tasks)

    async def _pop_task_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        if self.redis_client:
            raw_list = await self.redis_client.lrange(self.REDIS_KEY_PENDING, 0, -1)
            for item in raw_list:
                try:
                    data = json.loads(item)
                    if data.get("request_id") == request_id:
                        removed = await self.redis_client.lrem(self.REDIS_KEY_PENDING, 1, item)
                        if removed > 0:
                            return data
                except Exception:
                    pass
            return None
        else:
            async with self._in_memory_lock:
                for idx, task in enumerate(self._in_memory_pending_tasks):
                    if task.get("request_id") == request_id:
                        return self._in_memory_pending_tasks.pop(idx)
            return None

    @staticmethod
    def _dict_to_generate_request(d: Dict[str, Any]) -> queue_pb2.GenerateRequest:
        sp_dict = d.get("sampling_params", {})
        sp_proto = queue_pb2.SamplingParams(
            temperature=float(sp_dict.get("temperature", 1.0)),
            top_p=float(sp_dict.get("top_p", 1.0)),
            top_k=int(sp_dict.get("top_k", -1)),
            max_tokens=int(sp_dict.get("max_tokens", 64)),
            stop=sp_dict.get("stop", []),
        )
        return queue_pb2.GenerateRequest(
            request_id=d["request_id"],
            prompt=d.get("prompt", ""),
            sampling_params=sp_proto,
            priority=int(d.get("priority", 0)),
        )

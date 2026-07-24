# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration and End-to-End tests for Pull Worker Entrypoint."""

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from vllm.entrypoints.pull_worker.proto import queue_pb2, queue_pb2_grpc
from vllm.entrypoints.pull_worker.stat_logger import GrpcQueueWorkerStatLogger
from vllm.entrypoints.pull_worker.worker import (
    PullWorkerClient,
    map_finish_reason,
    map_proto_lora_request,
    map_proto_sampling_params,
)
from vllm.outputs import CompletionOutput, RequestOutput


def test_proto_mappings():
    """Test mapping helpers between gRPC proto and vLLM native types."""
    # SamplingParams
    proto_sp = queue_pb2.SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        max_tokens=100,
        stop=["\n", "END"],
    )
    native_sp = map_proto_sampling_params(proto_sp)
    assert native_sp.temperature == pytest.approx(0.8)
    assert native_sp.top_p == pytest.approx(0.95)
    assert native_sp.top_k == 40
    assert native_sp.max_tokens == 100
    assert native_sp.stop == ["\n", "END"]

    # LoRARequest
    proto_lora = queue_pb2.LoRARequest(
        lora_name="test-adapter",
        lora_id=1,
        lora_path="/path/to/lora",
    )
    native_lora = map_proto_lora_request(proto_lora)
    assert native_lora is not None
    assert native_lora.lora_name == "test-adapter"
    assert native_lora.lora_int_id == 1

    # FinishReason
    assert map_finish_reason("stop") == queue_pb2.STOP
    assert map_finish_reason("length") == queue_pb2.LENGTH
    assert map_finish_reason("abort") == queue_pb2.ABORT


class FakeQueueServerServicer(queue_pb2_grpc.PullWorkerServiceServicer):
    """Fake Queue Server Servicer for integration testing."""

    def __init__(self):
        self.registered_workers = {}
        self.statuses = []
        self.completed_chunks = []
        self.task_queue = asyncio.Queue()

    async def RegisterWorker(self, request, context):
        self.registered_workers[request.metadata.worker_id] = request.metadata
        return queue_pb2.RegisterWorkerResponse(
            worker_id=request.metadata.worker_id,
            session_token="test-session-token",
            registered_at_unix_s=123456789,
        )

    async def ReportStatus(self, request, context):
        self.statuses.append(request)
        return queue_pb2.ReportStatusResponse(acknowledged=True)

    async def StreamTasks(self, request_iterator, context):
        async for grant in request_iterator:
            if grant.slots_available > 0 and not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()
                    yield task
                except asyncio.QueueEmpty:
                    pass

    async def StreamOutput(self, request_iterator, context):
        async for chunk in request_iterator:
            self.completed_chunks.append(chunk)
        return queue_pb2.ReportStatusResponse(acknowledged=True)


@pytest.mark.asyncio
async def test_pull_worker_e2e_flow():
    """E2E Integration test verifying registration, PULL task execution, and streaming output."""
    # 1. Start Fake Queue Server
    servicer = FakeQueueServerServicer()
    server = grpc.aio.server()
    queue_pb2_grpc.add_PullWorkerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    queue_address = f"127.0.0.1:{port}"

    # Enqueue a dummy test task into server
    test_task = queue_pb2.GenerateRequest(
        request_id="test-req-1",
        prompt="Hello vLLM",
        sampling_params=queue_pb2.SamplingParams(temperature=0.7, max_tokens=10),
    )
    await servicer.task_queue.put(test_task)

    # 2. Mock AsyncLLM Engine
    mock_engine = MagicMock()
    mock_engine.model_config.model = "meta-llama/Llama-3-8B"
    mock_engine.model_config.max_model_len = 4096
    mock_engine.vllm_config.cache_config.block_size = 16
    mock_engine.vllm_config.cache_config.kv_cache_size_tokens = 65536
    mock_engine.vllm_config.parallel_config.world_size = 1
    mock_engine.vllm_config.parallel_config.data_parallel_size = 1

    async def fake_generate(prompt, sampling_params, request_id, **kwargs):
        # Yield two output chunks simulating token generation
        yield RequestOutput(
            request_id=request_id,
            prompt="Hello vLLM",
            prompt_token_ids=[1, 2],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(index=0, text=" World", token_ids=[3], cumulative_logprob=0.0, logprobs=None, finish_reason=None)
            ],
            finished=False,
        )
        yield RequestOutput(
            request_id=request_id,
            prompt="Hello vLLM",
            prompt_token_ids=[1, 2],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(index=0, text=" World!", token_ids=[3, 4], cumulative_logprob=0.0, logprobs=None, finish_reason="stop")
            ],
            finished=True,
        )

    mock_engine.generate = fake_generate
    mock_engine.abort = AsyncMock()

    # 3. Mock StatLogger
    mock_stat_logger = MagicMock()
    mock_stat_logger.get_latest_worker_status.return_value = queue_pb2.WorkerStatus(
        worker_id="worker-e2e-1",
        standard_metrics=queue_pb2.StandardMetrics(num_running_reqs=0, free_gpu_kv_blocks=100),
    )

    # 4. Start PullWorkerClient
    client = PullWorkerClient(
        worker_id="worker-e2e-1",
        queue_address=queue_address,
        engine=mock_engine,
        stat_logger=mock_stat_logger,
        max_concurrency=10,
    )

    await client.start()

    # Wait for task to be pulled and processed
    await asyncio.sleep(1.5)

    # 5. Assertions
    assert "worker-e2e-1" in servicer.registered_workers
    assert len(servicer.completed_chunks) >= 1

    # Check output text streamed back
    final_chunk = servicer.completed_chunks[-1]
    assert final_chunk.request_id == "test-req-1"
    assert final_chunk.finish_reason == queue_pb2.STOP

    # Cleanup
    await client.shutdown()
    await server.stop(grace=1.0)

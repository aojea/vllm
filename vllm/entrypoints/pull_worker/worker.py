# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM Disaggregated Pull-Based Queue Worker Entrypoint.

Starts a pull-worker client that connects to an external gRPC queue server,
reports status/metrics, pulls tasks based on engine capacity, and streams
generated outputs back to the queue server.
"""

import argparse
import asyncio
import signal
import sys
import uuid

import grpc
import uvloop

from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.pull_worker.proto import queue_pb2, queue_pb2_grpc
from vllm.entrypoints.pull_worker.stat_logger import GrpcQueueWorkerStatLogger
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def map_proto_sampling_params(
    proto_params: queue_pb2.SamplingParams,
) -> SamplingParams:
    """Map gRPC SamplingParams message to vLLM native SamplingParams."""
    return SamplingParams(
        temperature=proto_params.temperature if proto_params.temperature != 0.0 else 1.0,
        top_p=proto_params.top_p if proto_params.top_p != 0.0 else 1.0,
        top_k=proto_params.top_k if proto_params.top_k != 0 else -1,
        max_tokens=proto_params.max_tokens if proto_params.max_tokens != 0 else 16,
        stop=list(proto_params.stop) if proto_params.stop else None,
        presence_penalty=proto_params.presence_penalty,
        frequency_penalty=proto_params.frequency_penalty,
        repetition_penalty=proto_params.repetition_penalty if proto_params.repetition_penalty != 0.0 else 1.0,
        min_tokens=proto_params.min_tokens,
        n=proto_params.n if proto_params.n != 0 else 1,
        ignore_eos=proto_params.ignore_eos,
    )


def map_proto_lora_request(
    proto_lora: queue_pb2.LoRARequest | None,
) -> LoRARequest | None:
    """Map gRPC LoRARequest message to vLLM native LoRARequest."""
    if not proto_lora or not proto_lora.lora_name:
        return None
    return LoRARequest(
        lora_name=proto_lora.lora_name,
        lora_int_id=proto_lora.lora_id,
        lora_path=proto_lora.lora_path,
    )


def map_finish_reason(reason_str: str | None) -> queue_pb2.FinishReason:
    """Map string finish reason to gRPC FinishReason enum."""
    if not reason_str:
        return queue_pb2.FINISH_REASON_UNSPECIFIED
    r = str(reason_str).lower()
    if r == "stop":
        return queue_pb2.STOP
    elif r == "length":
        return queue_pb2.LENGTH
    elif r == "abort":
        return queue_pb2.ABORT
    elif r == "error":
        return queue_pb2.ERROR
    elif r == "repetition":
        return queue_pb2.REPETITION
    return queue_pb2.FINISH_REASON_UNSPECIFIED


class PullWorkerClient:
    """Pull Worker Client managing gRPC communication with the queue server

    and request execution via AsyncLLM.
    """

    def __init__(
        self,
        worker_id: str,
        queue_address: str,
        engine: AsyncLLM,
        stat_logger: GrpcQueueWorkerStatLogger,
        max_concurrency: int = 256,
    ):
        self.worker_id = worker_id
        self.queue_address = queue_address
        self.engine = engine
        self.stat_logger = stat_logger
        self.max_concurrency = max_concurrency

        self.channel: grpc.aio.Channel | None = None
        self.stub: queue_pb2_grpc.PullWorkerServiceStub | None = None

        self.active_tasks: dict[str, asyncio.Task] = {}
        self.stop_event = asyncio.Event()

    async def start(self) -> None:
        """Connect to queue server, register worker, and start pull loop."""
        logger.info(
            "Connecting Pull Worker %s to Queue Server at %s...",
            self.worker_id,
            self.queue_address,
        )

        self.channel = grpc.aio.insecure_channel(self.queue_address)
        self.stub = queue_pb2_grpc.PullWorkerServiceStub(self.channel)

        # 1. Register Worker with metadata
        metadata = queue_pb2.WorkerMetadata(
            worker_id=self.worker_id,
            model_name=self.engine.model_config.model,
            max_model_len=self.engine.model_config.max_model_len,
            block_size=self.engine.vllm_config.cache_config.block_size,
            kv_cache_size_tokens=getattr(
                self.engine.vllm_config.cache_config, "kv_cache_size_tokens", 0
            )
            or 0,
            world_size=self.engine.vllm_config.parallel_config.world_size,
            data_parallel_size=self.engine.vllm_config.parallel_config.data_parallel_size,
            vllm_version=VLLM_VERSION,
        )

        try:
            reg_resp = await self.stub.RegisterWorker(
                queue_pb2.RegisterWorkerRequest(metadata=metadata)
            )
            logger.info(
                "Worker registered successfully. Session token: %s",
                reg_resp.session_token,
            )
        except Exception as e:
            logger.error("Failed to register worker with Queue Server: %s", e)
            raise

        # 2. Start Background Status Reporter
        asyncio.create_task(self._report_status_loop())

        # 3. Start Task Pull Loop
        asyncio.create_task(self._task_pull_loop())

    async def _report_status_loop(self) -> None:
        """Periodically report worker status and metrics to queue server."""
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                if self.stub:
                    status = self.stat_logger.get_latest_worker_status(
                        self.worker_id
                    )
                    await self.stub.ReportStatus(status)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error in status report loop: %s", e)

    async def _task_pull_loop(self) -> None:
        """Credit-based bidirectional task pull loop."""
        while not self.stop_event.is_set():
            try:
                async def request_generator():
                    while not self.stop_event.is_set():
                        slots_available = max(
                            0, self.max_concurrency - len(self.active_tasks)
                        )
                        status = self.stat_logger.get_latest_worker_status(
                            self.worker_id
                        )
                        yield queue_pb2.TaskGrantRequest(
                            worker_id=self.worker_id,
                            slots_available=slots_available,
                            current_status=status,
                        )
                        await asyncio.sleep(0.5)

                assert self.stub is not None
                async for generate_request in self.stub.StreamTasks(
                    request_generator()
                ):
                    if self.stop_event.is_set():
                        break
                    # Dispatch task for execution
                    task = asyncio.create_task(
                        self._process_request(generate_request)
                    )
                    self.active_tasks[generate_request.request_id] = task
                    task.add_done_callback(
                        lambda t, req_id=generate_request.request_id: self.active_tasks.pop(
                            req_id, None
                        )
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in StreamTasks pull loop: %s. Retrying...", e)
                await asyncio.sleep(2.0)

    async def _process_request(
        self, req: queue_pb2.GenerateRequest
    ) -> None:
        """Process an incoming GenerateRequest via engine.generate and stream output."""
        request_id = req.request_id
        try:
            sampling_params = map_proto_sampling_params(req.sampling_params)
            lora_request = map_proto_lora_request(req.lora_request)

            prompt = req.prompt if req.prompt else list(req.prompt_token_ids)

            stream = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=req.priority,
            )

            async def output_chunk_generator():
                last_text_len = 0
                async for request_output in stream:
                    output = request_output.outputs[0]
                    new_text = output.text[last_text_len:]
                    last_text_len = len(output.text)

                    chunk = queue_pb2.RequestOutputChunk(
                        request_id=request_id,
                        new_token_ids=list(output.token_ids),
                        new_text=new_text,
                        finish_reason=map_finish_reason(output.finish_reason),
                        stop_reason=str(output.stop_reason) if output.stop_reason else "",
                    )
                    yield chunk

            assert self.stub is not None
            await self.stub.StreamOutput(output_chunk_generator())

        except Exception as e:
            logger.error("Error processing request %s: %s", request_id, e)
            if self.stub:
                err_chunk = queue_pb2.RequestOutputChunk(
                    request_id=request_id,
                    finish_reason=queue_pb2.ERROR,
                    error_message=str(e),
                )
                async def err_generator():
                    yield err_chunk
                try:
                    await self.stub.StreamOutput(err_generator())
                except Exception:
                    pass

    async def shutdown(self) -> None:
        """Gracefully shutdown the client and release resources."""
        logger.info("Shutting down PullWorkerClient %s...", self.worker_id)
        self.stop_event.set()

        for req_id, task in list(self.active_tasks.items()):
            task.cancel()
            await self.engine.abort(req_id)

        if self.channel:
            await self.channel.close()
        logger.info("PullWorkerClient shutdown complete.")


async def serve_pull_worker(args: argparse.Namespace) -> None:
    """Main serving function for pull worker entrypoint."""
    worker_id = args.worker_id or f"pull-worker-{uuid.uuid4().hex[:8]}"
    extended_metrics = set(
        [m.strip() for m in args.extended_metrics.split(",") if m.strip()]
    )

    logger.info("Starting vLLM Pull Worker %s", worker_id)
    logger.info("Queue Server Address: %s", args.queue_server_address)
    logger.info("Extended Metrics Enabled: %s", extended_metrics)

    # 1. Create Engine Args & Config
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    # 2. Instantiate Custom Stat Logger
    stat_logger = GrpcQueueWorkerStatLogger(
        vllm_config=vllm_config,
        extended_metrics_config=extended_metrics,
    )

    # 3. Instantiate AsyncLLM Engine
    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        stat_loggers=[stat_logger],
        enable_log_requests=args.enable_log_requests,
        disable_log_stats=args.disable_log_stats,
    )

    # 4. Instantiate Pull Worker Client
    client = PullWorkerClient(
        worker_id=worker_id,
        queue_address=args.queue_server_address,
        engine=async_llm,
        stat_logger=stat_logger,
        max_concurrency=args.max_concurrency,
    )

    # 5. Handle Signal Shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await client.start()
        await stop_event.wait()
    finally:
        await client.shutdown()
        async_llm.shutdown()


def main():
    """Main CLI entrypoint for python -m vllm.entrypoints.pull_worker.worker."""
    parser = FlexibleArgumentParser(
        description="vLLM Disaggregated Pull-Based Queue Worker",
    )

    parser.add_argument(
        "--queue-server-address",
        type=str,
        required=True,
        help="Address of the central gRPC Queue Server (host:port)",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default="",
        help="Unique identifier for this pull worker instance",
    )
    parser.add_argument(
        "--extended-metrics",
        type=str,
        default="",
        help="Comma-separated list of extended metrics to report (e.g. prefix_cache,perf_latency,all)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=256,
        help="Maximum concurrent request slots for task pulling",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    try:
        uvloop.run(serve_pull_worker(args))
    except Exception as e:
        logger.exception("Pull Worker failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

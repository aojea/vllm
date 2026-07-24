# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Executable Queue Server for vLLM Disaggregated Pull Queue."""

import argparse
import asyncio
import logging
import signal
import sys

import grpc
import uvloop

from examples.heuristic_pull_router.framework import PullQueueFramework
from examples.heuristic_pull_router.policy import PrefixAndSLAAwarePolicy
from vllm.entrypoints.pull_worker.proto import queue_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("queue_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM Disaggregated Pull Queue Server with Pluggable Routing Policy"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind gRPC server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind gRPC server (default: 50051)",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default=None,
        help="Optional Redis host for persistent distributed state storage",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--prefix-weight",
        type=float,
        default=1000.0,
        help="Policy weight for prefix-cache locality matches (default: 1000.0)",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=500.0,
        help="Policy weight for SLA priority tier (default: 500.0)",
    )
    parser.add_argument(
        "--age-weight",
        type=float,
        default=10.0,
        help="Policy weight per second of waiting age (default: 10.0)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # 1. Instantiate Policy
    policy = PrefixAndSLAAwarePolicy(
        prefix_weight=args.prefix_weight,
        priority_weight=args.priority_weight,
        age_weight=args.age_weight,
    )

    # 2. Instantiate Framework
    framework = PullQueueFramework(
        policy=policy,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )
    await framework.initialize()

    # 3. Create gRPC Server
    server = grpc.aio.server()
    queue_pb2_grpc.add_PullWorkerServiceServicer_to_server(framework, server)
    listen_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(listen_addr)

    logger.info("Starting Disaggregated Pull Queue Server at %s", listen_addr)
    await server.start()

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received. Stopping server...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    await stop_event.wait()
    await server.stop(grace=5.0)
    logger.info("Queue Server stopped gracefully.")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())

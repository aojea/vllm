# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Traffic Generator for testing vLLM Disaggregated Pull Queue."""

import argparse
import asyncio
import logging
import random
import time

from examples.heuristic_pull_router.policy import compute_prompt_prefix_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("simulate_traffic")

PREFIX_WORKLOADS = {
    "CODE_GEN": "SYSTEM PROMPT: You are an expert Python software engineer specializing in high-performance concurrent async code. ",
    "LEGAL_QA": "SYSTEM PROMPT: You are a legal compliance analyzer tasked with reviewing contract terminology and regulatory standards. ",
    "MATH_SOLVER": "SYSTEM PROMPT: You are a mathematical reasoning assistant. Provide step-by-step proofs for complex algebraic problems. ",
}

SAMPLE_PROMPTS = [
    ("CODE_GEN", "Write an async function to compute Fibonacci numbers."),
    ("CODE_GEN", "Implement a thread-safe LRU cache in Python."),
    ("LEGAL_QA", "Summarize indemnification clauses under California commercial law."),
    ("LEGAL_QA", "What are key compliance risks under GDPR Article 17?"),
    ("MATH_SOLVER", "Prove that the sum of the first N odd numbers is N squared."),
    ("MATH_SOLVER", "Calculate the derivative of f(x) = x^3 * sin(x)."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic Generator for vLLM Disaggregated Pull Queue"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:50051",
        help="Queue server gRPC address (default: 127.0.0.1:50051)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=12,
        help="Number of requests to generate (default: 12)",
    )
    parser.add_argument(
        "--vip-ratio",
        type=float,
        default=0.25,
        help="Ratio of VIP high-priority tenant requests (default: 0.25)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logger.info("Initializing traffic simulation with %d requests...", args.num_requests)

    # Note: In an end-to-end integration deployment, clients submit requests
    # into the queue server via Redis or an HTTP API proxy.
    print("\n=== Simulated Workload Tasks Generated ===")
    for i in range(args.num_requests):
        domain, query = random.choice(SAMPLE_PROMPTS)
        prefix_str = PREFIX_WORKLOADS[domain]
        full_prompt = prefix_str + query
        p_hash = compute_prompt_prefix_hash(full_prompt)

        is_vip = random.random() < args.vip_ratio
        priority = 10 if is_vip else 0
        tier = "VIP" if is_vip else "Standard"

        print(
            f"[{i+1:02d}] RequestID: req-{i+1:03d} | Domain: {domain:11s} | "
            f"PrefixHash: {p_hash:#010x} | Tier: {tier:8s} (Priority: {priority:2d})"
        )
        print(f"     Prompt: '{query}'")

    print("\nSimulation payload successfully generated!")
    print(
        "To test with live vLLM pull_worker instances, run queue_server.py and launch workers with:"
    )
    print(
        "  python -m examples.heuristic_pull_router.queue_server --port 50051"
    )


if __name__ == "__main__":
    asyncio.run(main())

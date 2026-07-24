# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Local conftest for pull_worker tests."""

import pytest


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """Disable global accelerator memory cleanup in CPU testing environment."""
    return False


pytest_plugins = ()

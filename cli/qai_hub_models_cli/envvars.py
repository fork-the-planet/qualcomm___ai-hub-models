# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""If set, forces the CLI to act as if it was running as this AI Hub Models version."""

import os

FORCE_VERSION_ENVVAR = "QAIHM_CLI_FORCE_VERSION"

"""
If set, FOR ANY RELEASE, forces the CLI to load manifest files from this path.
In practice, this means all releases will display the same information.
"""
FORCE_MANIFEST_ROOT_ENVVAR = "QAIHM_CLI_MANIFEST_ROOT"

"""
If set to "1", enables verbose exceptions. By default ("0"), the CLI will swallow the traceback.
"""
VERBOSE_EXCEPTIONS_ENVVAR = "QAIHM_CLI_VERBOSE_EXCEPTIONS"

"""
If set to "1", allows the CLI to use the internal (private) S3 releases.
Requires valid AWS credentials (profile: qaihm) and the [internal] extra.
"""
USE_INTERNAL_RELEASES_ENVVAR = "QAIHM_CLI_USE_INTERNAL_RELEASES"

"""
If set, overrides the default AWS session duration (seconds) written into
``~/.saml2aws`` by ``validate_credentials``. Clamped to [3600, 28800].
"""
AWS_SESSION_DURATION_ENVVAR = "QAIHM_AWS_SESSION_DURATION"


def bool_envvar_value(envvar: str, default: bool = False) -> bool:
    return os.environ.get(envvar, "1" if default else "0").lower() in [
        "1",
        "true",
        "yes",
    ]

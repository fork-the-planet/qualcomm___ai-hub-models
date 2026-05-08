# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import shlex

import qai_hub as hub


def extract_job_options(job: hub.Job) -> dict[str, str | bool]:
    """
    Get a dictionary of all options passed for this job.

    Options that are not passed explicitly in the workbench job options (that Hub will treat as defaults) are not included in the dictionary.

    Parameters
    ----------
    job
        The job from which to extract options.

    Returns
    -------
    options : dict[str, str | bool]
        Dictionary of all options passed for this job.

    Examples
    --------
    If a hub job is submitted with options "--qairt_version 2.33 --dequantize_outputs --dict_input='w=x;y=z'"
    Then the returned dict would be:

    .. code-block:: python

        {
            "qairt_version": "2.33",
            "dequantize_outputs": True,
            "dict_input": "w=x;y=z"
        }
    """
    out = {}

    model_options = shlex.split(job.options.strip())
    for i in range(len(model_options)):
        option = model_options[i]
        if option.startswith("--"):
            value: str | bool
            if "=" in option:
                # Handle args of form "--blah=blah"
                #
                # If the option starts with '--' and has an =, then it must be in the format --x=y.
                # If the option was in the format --x y=x, then it would be parsed as two different options by shlex.
                # So we can safely split this option on the first '=' in the string to get the option key and value.
                key, value = option.split("=", maxsplit=1)
                if (value.startswith("'") and value.endswith("'")) or (
                    value.startswith('"') and value.endswith('"')
                ):
                    value = value[1 : len(value) - 1]
            elif i == len(model_options) - 1 or model_options[i + 1].startswith("--"):
                # Either:
                #   - this is the last arg and has no value
                #   - this --arg is immediately followed by another --arg
                # Therefore it must be a boolean. All Hub booleans are true if explicitly passed as an option.
                key = option
                value = True
            else:
                # This is a standard --key value pair.
                key = option
                value = model_options[i + 1]
            # Strip "--" from arg name.
            out[key[2:]] = value

    return out

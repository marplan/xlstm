# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel

import os
import shlex
import logging
from typing import Sequence

import time
import random

import torch
from torch.utils.cpp_extension import include_paths as _include_paths, load as _load

LOGGER = logging.getLogger(__name__)


def defines_to_cflags(
    defines: dict[str, int | str] | Sequence[tuple[str, str | int]] = (),
):
    cflags: list[str] = []
    if isinstance(defines, dict):
        items = [(key, val) for key, val in defines.items()]
    else:
        items = list(defines)
    for key, val in items:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)

if torch.cuda.is_available():
    os.environ["CUDA_LIB"] = os.path.join(
        os.path.split(_include_paths(device_type="cuda")[-1])[0], "lib"
    )


EXTRA_INCLUDE_PATHS = (
    tuple(os.environ["XLSTM_EXTRA_INCLUDE_PATHS"].split(":"))
    if "XLSTM_EXTRA_INCLUDE_PATHS" in os.environ
    else ()
)


def _extra_cuda_flags_from_env() -> tuple[str, ...]:
    raw = os.environ.get("XLSTM_EXTRA_CUDA_FLAGS")
    if raw is None:
        return ()
    return tuple(flag for flag in shlex.split(raw) if flag)


def _with_unique_flags(*groups: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for flag in group:
            if flag in seen:
                continue
            seen.add(flag)
            ordered.append(flag)
    return ordered


DEFAULT_EXTRA_CUDA_FLAGS: tuple[str, ...] = ("-static-global-template-stub=false",)
if "CONDA_PREFIX" in os.environ:
    # This enforces adding the correct include directory from the CUDA installation via torch. If you use the system
    # installation, you might have to add the cflags yourself.
    from pathlib import Path
    from packaging import version
    import glob

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        matching_dirs = glob.glob(f"{os.environ['CONDA_PREFIX']}/targets/**", recursive=True)
        EXTRA_INCLUDE_PATHS = (
            EXTRA_INCLUDE_PATHS
            + tuple(map(str, (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("**/include/")))[
                :1
            ]
        )


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {"float": "f", "__half": "h", "__nv_bfloat16": "b", "true": "1", "false": "0"}
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        # *(f"-I{path}" for path in EXTRA_INCLUDE_PATHS)
    ]
    for eip in EXTRA_INCLUDE_PATHS:
        extra_cflags.append("-isystem")
        extra_cflags.append(eip)

    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],
        "extra_cflags": [*extra_cflags],
        "extra_cuda_cflags": [
            # "-gencode",
            # "arch=compute_70,code=compute_70",
            # "-dbg=1",
            '-Xptxas="-v"',
            "-gencode",
            "arch=compute_80,code=compute_80",
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            *extra_cflags,
            *extra_cuda_cflags,
            *_with_unique_flags(DEFAULT_EXTRA_CUDA_FLAGS, _extra_cuda_flags_from_env()),
        ],
    }
    print(myargs)
    myargs.update(**kwargs)
    # add random waiting time to minimize deadlocks because of badly managed multicompile of pytorch ext
    time.sleep(random.random() * 10)
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod

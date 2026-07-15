# Qualcomm AI Hub Models CLI

A command-line tool for browsing and downloading Qualcomm® AI Hub Models.

- **Browse** and filter the model catalog.
- **Inspect** a model's metadata, performance, and numerics.
- **Download** ready-to-run model assets for a specific runtime and device.
- **Explore** the devices, chipsets, and runtimes supported by each release.

## Installation

The CLI is lightweight. With dependencies, it takes only a few MB on disk.

Install from pypi:

```bash
pip install qai_hub_models_cli
```

This installs the `qai-hub-models` console entry point:

```bash
qai-hub-models --help
```

## Quick start

```bash
# Find a pre-compiled asset and download it
qai-hub-models models                                  # browse the catalog
qai-hub-models info mobilenet_v2                        # details + download options
qai-hub-models fetch mobilenet_v2 --runtime tflite --precision float   # download it
```

Every command prints follow-up suggestions, so you can usually discover the
next step from the output itself.

## Commands

Run `qai-hub-models <command> --help` (e.g. `qai-hub-models fetch --help`) for
the full flag list of any command.

### Models

| Command    | Purpose                                                   | Example                                              |
| ---------- | --------------------------------------------------------- | ---------------------------------------------------- |
| `fetch`    | Download a model, or list options with `-i/--info` *      | `qai-hub-models fetch mobilenet_v2 -r tflite -p float` |
| `info`     | Show metadata and download options for a model            | `qai-hub-models info mobilenet_v2`                  |
| `perf`     | Show a model's performance metrics *                      | `qai-hub-models perf mobilenet_v2`                  |
| `numerics` | Show a model's accuracy metrics *                         | `qai-hub-models numerics mobilenet_v2`              |
| `find`     | Search past releases for a matching asset *               | `qai-hub-models find mobilenet_v2 -s qairt=2.45` |

\* These commands accept filter flags to be passed, to narrow their results — see
[Filtering](#filtering).

### Customized Models (export from source)

| Command    | Purpose                                                   | Example                                              |
| ---------- | --------------------------------------------------------- | ---------------------------------------------------- |
| `export` | Export a model to a Qualcomm runtime via AI Hub Workbench | `qai-hub-models export mobilenet_v2 -r tflite -p float -d "Samsung Galaxy S25 (Family)"` |
| `evaluate` | Evaluate a model's accuracy on a dataset via AI Hub Workbench | `qai-hub-models evaluate mobilenet_v2 -r tflite -p float -d "Samsung Galaxy S25 (Family)"` |

`export` and `evaluate` require the full `qai_hub_models` package (`pip install qai_hub_models`).

### Catalog

| Command    | Purpose                                                   | Example                                              |
| ---------- | --------------------------------------------------------- | ---------------------------------------------------- |
| `models`   | List all available models *                               | `qai-hub-models models --domain "Computer Vision"`  |
| `devices`  | List all supported devices *                              | `qai-hub-models devices`                            |
| `chipsets` | List all supported chipsets *                             | `qai-hub-models chipsets`                           |
| `runtimes` | List all runtimes a model can be compiled to              | `qai-hub-models runtimes`                           |
| `versions` | List AI Hub Models versions supported by this CLI         | `qai-hub-models versions`                           |

\* These commands accept filter flags to be passed, to narrow their results — see
[Filtering](#filtering).


## Common flags

These flags are shared across most commands:

| Flag               | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `-h`, `--help`     | Shows all possible flags for the command, and exit                |
| `-v`, `--version`  | Target a specific release (e.g. `-v 0.45.0`). Defaults to the version matching this CLI install |
| `-q`, `--quiet`    | Machine-readable output: plain lists for listing commands, just the result path for `fetch` |

Some filters and table columns require a recent release — the CLI tells you when
one isn't available for the targeted version.

## Environment variables

| Variable | Description |
| --- | --- |
| `QAIHM_AWS_SESSION_DURATION` | Overrides the AWS session duration (seconds) written into `~/.saml2aws` by `validate_aws_credentials`. Clamped to `[3600, 28800]` (1h–8h). Useful for long-running headless callers whose runs exceed the 1h default. Only applies when using the `[internal]` extra. |

Set before running `validate_aws_credentials`:

```bash
export QAIHM_AWS_SESSION_DURATION=28800
validate_aws_credentials
```

## Filtering

The starred commands above accept these filter flags (run
`qai-hub-models <command> --help` for the full, per-command set). A record
matches if it satisfies the given value(s):

| Flag                  | Filters by                                                  |
| --------------------- | ----------------------------------------------------------- |
| `-r`, `--runtime`     | Runtime name (see `qai-hub-models runtimes`).               |
| `-p`, `--precision`   | Precision (e.g. `float`, `w8a8`).                           |
| `-c`, `--chipset`     | Chipset name (see `qai-hub-models chipsets`).               |
| `-d`, `--device`      | Device name (see `qai-hub-models devices`). Mutually exclusive with `--chipset`. |
| `-s`, `--sdk-version` | SDK/tool version, `tool=version` syntax (e.g. `qairt=2.20`). Use `--help` to see valid SDK names. |

Most filters take multiple values and can be repeated; the catalog (`models`)
also supports `--domain`, `--use-case`, `--quantized`, `--llm`, `--aot`/`--jit`,
and `-t`/`--tag`.

```bash
qai-hub-models perf mobilenet_v2 -r qnn -c qualcomm-snapdragon-8gen3
qai-hub-models models --domain "Computer Vision" --quantized
```

## Finding assets in past releases

When the current release no longer ships an asset you need, `find` searches
released versions — newest first — for one matching the same filters `fetch`
accepts, and reports the release(s) that have it:

```bash
# Newest release with tflite MobileNet-v2 assets that were tested with QAIRT 2.45
qai-hub-models find mobilenet_v2 -r tflite -s qairt=2.45

# Every matching release, not just the newest
qai-hub-models find mobilenet_v2 -r qnn -c qualcomm-snapdragon-8gen3 --all
```

Each match is printed with its download table and a ready-to-run `fetch` command
pinned to that release (`-v <version>`). Add `-q`/`--quiet` to print just the
matching version numbers, one per line.

## Python API

### Downloading models

Downloads can also be driven from Python via `qai_hub_models_cli.fetch`. This is
the same code path the `fetch` command uses.

```python
from qai_hub_models_cli.fetch import fetch, get_asset_url

# Download an asset and return the path on disk (extracts the zip by default).
path = fetch(
    model="mobilenet_v2",
    runtime="tflite",
    precision="float",
    output_dir="./assets",
    extract=True,
)
print(path)

# Device-specific (AOT-compiled) runtimes need a chipset or device.
path = fetch(
    model="mobilenet_v2",
    runtime="qnn",
    precision="w8a8",
    chipset="qualcomm-snapdragon-8gen3",
    output_dir="./assets",
)

# Resolve the download URL without downloading.
url = get_asset_url(
    model="mobilenet_v2", runtime="tflite", precision="float"
)
```

### Reading metadata

The same metadata behind the listing commands is available as protobuf objects.
Each getter takes a model ID (or display name) and an optional `version`, and
results are cached:

```python
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.perf import get_model_perf
from qai_hub_models_cli.proto_helpers.numerics import get_model_numerics
from qai_hub_models_cli.proto_helpers.manifest import get_manifest, get_manifest_entry
from qai_hub_models_cli.proto_helpers.platform import get_platform
from qai_hub_models_cli.proto_helpers.release_assets import get_model_release_assets

info = get_model_info("mobilenet_v2")          # ModelInfo: name, description, license, tags, …
print(info.name, info.domain)

perf = get_model_perf("mobilenet_v2")          # ModelPerf: per-device performance metrics
numerics = get_model_numerics("mobilenet_v2")  # ModelNumerics: per-device accuracy metrics
assets = get_model_release_assets("mobilenet_v2")  # ModelReleaseAssets: available downloads

manifest = get_manifest()                      # ReleaseManifest: every model in the release
for entry in manifest.models:
    print(entry.id, entry.display_name)

platform = get_platform()                      # PlatformInfo: supported devices, chipsets, runtimes
```

Each getter's module also provides a matching `filter_*` helper that applies the
same filtering the CLI flags use:

```python
from qai_hub_models_cli.proto_helpers.perf import filter_perf
from qai_hub_models_cli.proto_helpers.numerics import filter_numerics
from qai_hub_models_cli.proto_helpers.release_assets import filter_release_assets
from qai_hub_models_cli.proto_helpers.platform import filter_devices, filter_chipsets
```

### Searching past releases

`qai_hub_models_cli.find` backs the `find` command. `find_matching_releases`
searches releases (newest-first) and returns `(version, matching_assets)` pairs;
`find_in_version` checks a single release and returns the matching assets or
`None`.

```python
from qai_hub_models_cli.find import find_matching_releases, find_in_version

# Newest release with a matching asset (first_only stops at the first hit).
hits = find_matching_releases(
    "mobilenet_v2", runtime="tflite", precision="float", first_only=True
)
for version, assets in hits:
    print(version, len(assets.assets))

# Check one specific release.
from packaging.version import Version
assets = find_in_version("mobilenet_v2", Version("0.52.0"), runtime="tflite")
```

## See also

- **Collection of AI Hub Models:** <https://aihub.qualcomm.com/models>
- **Source & model export scripts:** <https://github.com/qualcomm/ai-hub-models>
- **Request a new model:** <https://github.com/qualcomm/ai-hub-models/issues>

> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [PiperTTS-DE: High-quality text-to-speech library](https://aihub.qualcomm.com/models/pipertts_de)

PiperTTS is a high-quality multi-lingual text-to-speech library.

This is based on the implementation of PiperTTS-DE found [here](https://github.com/rhasspy/piper).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/pipertts_de).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install System-Level Dependencies
#### Linux
```bash
sudo apt install libportaudio2
```

### 2. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.12 is supported.
pip install git+https://github.com/rhasspy/piper.git#subdirectory=src/python --no-deps
pip install "qai-hub-models[pipertts-de]"
```

### 3. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.pipertts_de.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.pipertts_de.export --device "Samsung Galaxy S25 (Family)"
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of PiperTTS-DE can be found
  [here](https://github.com/rhasspy/piper/blob/master/LICENSE.md).

## References
* [PiperTTS High-quality Multi-lingual Multi-accent Text-to-Speech](https://github.com/rhasspy/piper)
* [Source Model Implementation](https://github.com/rhasspy/piper)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

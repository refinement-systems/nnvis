# NNVis Build and Run Instructions

NNVis visualizes HuggingFace Neural Network architectures right from your browser using Three.js and a custom Python extraction script.

## Pre-requisites
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Safari)
- HuggingFace local models (Safetensors and ONNX formats needed. Use `huggingface-cli download` to fetch it to a local folder, with `--include="*.safetensors" "*.onnx"`)

## 1. Setup Python Environment
Create and activate an isolated virtual environment to manage dependencies for the extraction script.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Extract Conceptual Layer Data
The python extraction scripts analyze the ONNX and SafeTensors to detect conceptual layers, bounding mappings, and the final 3D node layout, which is saved into `model_summary.json`. 

```bash
# Example syntax: python extract.py path/to/model-directory
python extract.py models/MoritzLaurer_mDeBERTa-v3-base-mnli-xnli
```
> [!NOTE] 
> This assumes you have already downloaded the model files from HuggingFace to the specified directory. The script expects the `model.safetensors` file inside the root, and the `model.onnx` file inside the `/onnx/` subdirectory.

## 3. Viewing the App
The frontend is built with pure Vanilla HTML/CSS/JS with ES Modules (for Three.js). Due to CORS policies when dynamically loading local modules (`app.js` and dependencies), you must run it with a local web server rather than just opening the `.html` file directly from the filesystem.

You can spin one up instantly with python:
```bash
python3 -m http.server 8000
```
Then navigate to `http://localhost:8000/index.html` in your browser. From there, you can drag and drop your generated `model_summary.json` file into the UI, or, if you test the included model, simply navigate to `http://localhost:8000/index.html?test=true`.

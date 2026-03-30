# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
# AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import argparse
import os
import re
from huggingface_hub import snapshot_download

def sanitize_model_name(model_id: str) -> str:
    """Replace slashes and other non-filename characters with underscores."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', model_id)

def main():
    parser = argparse.ArgumentParser(description="Download model files for visualization.")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)")
    args = parser.parse_args()
    
    sanitized_id = sanitize_model_name(args.model_id)
    out_dir = os.path.join("models", sanitized_id)
    
    print(f"Downloading {args.model_id} to {out_dir}...")
    
    # We only download the necessary files as outlined to build the visualization.
    allow_patterns = [
        "config.json",
        "*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",
        "onnx/model.onnx"
    ]
    
    snapshot_download(
        repo_id=args.model_id,
        allow_patterns=allow_patterns,
        local_dir=out_dir,
        # Set this to False to copy actual files instead of symlinking to the cache 
        # (makes it easier to browse).
        local_dir_use_symlinks=False
    )
    print("Download complete!")

if __name__ == "__main__":
    main()

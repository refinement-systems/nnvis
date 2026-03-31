from transformers import AutoConfig, AutoModelForSequenceClassification
import torch

model_dir = "/path/to/model"

config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    config=config,
    local_files_only=True
)
model.eval()

input_ids = torch.ones((1, 16), dtype=torch.long)
attention_mask = torch.ones((1, 16), dtype=torch.long)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "reexported.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamo=True,
)


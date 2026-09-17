import json
import tyro
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(adapter_path: str, output_path: str, base_model: str = ""):
    if not base_model:
        with open(f"{adapter_path}/adapter_config.json") as f:
            base_model = json.load(f)["base_model_name_or_path"]

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)
    print(f"Merged -> {output_path}")


if __name__ == "__main__":
    tyro.cli(main)

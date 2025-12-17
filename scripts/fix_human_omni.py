import transformers
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniForConditionalGeneration


def fix_human_omni():
    # human_omni_path = "PhilipC/HumanOmniV2"
    human_omni_path = "ddvd233/OmniSapiens-7B-RL"
    human_omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        human_omni_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    parent_model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    # merge parameters from human_omni_model to parent_model
    for name, param in human_omni_model.named_parameters():
        parent_name = "thinker." + name
        if parent_name in parent_model.state_dict():
            if param.shape == parent_model.state_dict()[parent_name].shape:
                parent_model.state_dict()[parent_name].copy_(param)

            else:
                print(f"Shape mismatch for {parent_name}: {param.shape} vs {parent_model.state_dict()[parent_name].shape}")
        else:
            print(f"Parameter {parent_name} not found in parent model")

    # Upload the new model to Hugging Face
    parent_model.push_to_hub("ddvd233/OmniSapiens-7B-RL-Full")


if __name__ == "__main__":
    fix_human_omni()
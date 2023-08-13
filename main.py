from modelling_RW import RWConfig
from configuration_RW import RWForCausalLM

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer
)
from transformers.utils import logging

import torch

def main():
    logging.set_verbosity_info()

    # Loading local config, model, and tokenizer classes with pretrained configs, is this the correct way to do it?
    pretrained_config_path = "falcon7b-pretrained-configs/config.json"
    falcon_config = RWConfig.from_pretrained(pretrained_config_path)
    print(f"Pre-trained falcon-7b config: {falcon_config}")

    falcon_model = RWForCausalLM.from_pretrained(
        pretrained_config_path,
        config=falcon_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"Pretrained falcon model: {falcon_model}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_config_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pre-trained tokenizer: {tokenizer}")

    inputs = tokenizer("What's the best way to divide a pizza between three people?", return_token_type_ids=False, return_tensors="pt").to("cuda")
    
    outputs = falcon_model.generate(**inputs, max_length=50, return_dict_in_generate=True)
    print(outputs)


if __name__ == "__main__":
    main()
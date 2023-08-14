from configuration_RW import RWConfig
#from modelling_RW import RWForCausalLM
from model_rotary import FalconRotaryForCausalLM


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    pipeline
)
from transformers.utils import logging

import torch


def load_falcon_rotary(falcon_config, model_name):
    model = FalconRotaryForCausalLM.from_pretrained(
        model_name,
        config=falcon_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"Pretrained falcon model with Rotary: {model}")
    return model

def load_falcon_alibi(falcon_config, model_name):
    model = FalconAlibiForCausalLM.from_pretrained(
        model_name,
        config=falcon_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"Pretrained falcon model with Alibi: {model}")
    return model

def main():
    logging.set_verbosity_info()

    # Loading local config, model, and tokenizer classes with pretrained configs, is this the correct way to do it?
    pretrained_config_path = "./falcon7b-pretrained-configs/config.json"
    model_name = "tiiuae/falcon-7b"

    # Get local config.json, same as from HF. 
    falcon_config = RWConfig.from_pretrained(pretrained_config_path)
    
    # Downloads model weights from HF repo and instanties a model from the local classes
    # TODO Download a specified checkpoint/commit OR download latest pre trained config from HF
    model = load_falcon_rotary(falcon_config, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pre-trained tokenizer: {tokenizer}")
    
    #simpleinput(model, tokenizer)
    simplepipe(model,tokenizer)

def simpleinput(model, tokenizer):
    inputs = tokenizer("What's the best way to divide a pizza between three people?", return_token_type_ids=False, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_length=500, return_dict_in_generate=True)

    decoded = tokenizer.decode(outputs.sequences[0])
    print(decoded)
    """ 
    What's the best way to divide a pizza between three people?
    I'm going to a pizza place with two friends and we're all going to order a pizza. I'm not sure what kind of pizza we're going to get, but I'm thinking of getting a small pizza with a few toppings. I'm not sure what kind of pizza we're going to get, but I'm thinking of getting a small pizza with a few toppings.
    I'm not sure what kind of pizza we're going to get, but I'm thinking of getting a small pizza with a few toppings.
    I'm not sure what kind of pizza we're going to get, but I'm thinking of getting a small pizza with a few toppings.
    ... repeating
    """
    # So this is garbage, but the simplepipe is ok, why


def simplepipe(model, tokenizer):
    pl = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    pizza = "What's the best way to divide a pizza between three people?"
    """
    Result: What's the best way to divide a pizza between three people?
    It's a common dilemma and the most sensible way to do it is to use a ruler.
    The only other method would be cutting up the pizza but that's not going to give you an even division and will leave you with less of the tasty stuff.
    The ruler was the preferred method of a maths teacher who recently shared a picture of a 'divider' she found in the school staff room.
    She said her colleague was "a very precise person" so they must have been doing the maths when they designed the tool.
    The image has been shared on Reddit where users were split as to which method was fairest - but many had come to a firm conclusion in favour of using the ruler.
    One wrote: "The best way to split a pizza is with a ruler, you know you have the right amount of each bit."
    """ 
    # much better than simpleinput, is it because of sampling and top-k perhaps?
    
    giff = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
    """
    Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
    Daniel: Hello, Girafatron!
    Girafatron: Hello, Daniel.
    Daniel: What’s up?
    Girafatron: Not much. Just trying to think of some way to celebrate my love of the giraffe.
    Daniel: What would you like to do?
    Girafatron: Well, I was thinking of throwing myself a big party.
    Daniel: What sort of party?
    Girafatron: I think it would be a very giraffish sort of party.
    Daniel: I would think that would be hard.
    Girafatron: Not at all! Just think of it! All giraffes! And all the giraffe things! It would be the
    """

    sequences = pl(
        giff,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")



if __name__ == "__main__":
    main()
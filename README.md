# falcon-llm

*A fork of https://huggingface.co/tiiuae/falcon-7b commit 378337427557d1df3e742264a2901a49f25d4eb1 without model weights*

Several issues are being discussed in the relevant hf repos, these repos also have several open PRs which address these issues, this repo will investigate these issues and PRs, and suggest further issues, to hopefully get an optimal working model. There also seems to be redundant code that is not used in the pre-trained model, so this repo should also simplify the code for increased readability and extendability.

## Big changes compared to original
- split model code in modelling_rw to model_rotary and model_alibi (Changing the name of RWforCasualLM to FalconRotaryforCasualLM or FalconAlibiForCasualLM)

## RWModel (Falcon) architecture

![Alt text](./diagrams/RWModel_architecture.png "RWModel/Falcon architecture")
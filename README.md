# falcon-llm

*A fork of https://huggingface.co/tiiuae/falcon-7b commit 378337427557d1df3e742264a2901a49f25d4eb1 without model weights*

This repository is primarily focused on refining and optimizing the pre-trained Falcon models and codebase. It's crucial to emphasize that my modifications are not intended to be used to replicate Falcon-like models from scratch. For example, there is a lot of configuration options in the existing codebase that is only relevant for pretraining and not relevant for use with the pretrained models. 

Goals:

- Enhance the codebase specifically for leveraging pre-trained models more efficiently.
- Simplify the code to improve readability and understandability for users and developers.
- Investigate and address various issues and PRs present in the related HF repositories.
- Identify and eliminate any redundant code that doesn't serve the primary purpose of the repo.

## RWModel (Falcon) architecture

![Alt text](./diagrams/RWModel_architecture.png "RWModel/Falcon architecture")
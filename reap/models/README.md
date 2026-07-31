# Models

Plant DNABERT-6mer backbone:

```text
models/pretrained/plant-dnabert-6mer/
```

REAP fold weights:

```text
models/finetuned/fold1/merged/
models/finetuned/fold2/merged/
models/finetuned/fold3/merged/
models/finetuned/fold4/merged/
models/finetuned/fold5/merged/
```

Each `merged/` directory contains:

```text
config.json
pytorch_model.bin
regularized_state.pt
special_tokens_map.json
tokenizer_config.json
vocab.txt
```

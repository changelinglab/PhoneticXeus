# Contributing to PhoneticXeus

## Project Structure

```
configs/                     <- Hydra configs
  callbacks/                    <- Callbacks configs
  data/                         <- Data configs
  debug/                        <- Debugging configs
  experiment/                   <- Experiment configs
    train/                        <- Training experiments
    inference/                    <- Inference experiments
  extras/                       <- Extra utilities configs
  hydra/                        <- Hydra configs
  local/                        <- Local configs (not versioned)
  logger/                       <- Logger configs
  model/                        <- Model configs
  paths/                        <- Project paths configs
  trainer/                      <- Trainer configs
  main.yaml                    <- Main config

scripts/                     <- SLURM batch scripts and shell helpers
src/                         <- Source code
  core/                         <- Task orchestration, tokenizer, distributed inference
  data/                         <- DataModules and dataset classes
  metrics/                      <- Evaluation metrics (phone recognition)
  model/                        <- Model architectures (powsm, wav2vec2, whisper, etc.)
  recipe/                       <- Recipe modules
    phone_recognition/             <- Phone recognition model + local analysis scripts
  utils/                        <- Utility scripts
  main.py                      <- Entry point
tests/                       <- Tests
```

## Workflow

1. Write or modify experiment configs in `configs/experiment/train/` or
   `configs/experiment/inference/`.
2. Run training:
   ```bash
   python src/main.py experiment=train/ipapack_xeuspr
   ```
3. Run inference:
   ```bash
   python src/main.py experiment=inference/powsmpreval data.dataset_name=doreco
   ```

See [docs/running_inference.md](docs/running_inference.md) for the distributed
inference workflow.

## Best Practices

<details>
<summary><b>Use automatic code formatting</b></summary>

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Reformat all files:

```bash
pre-commit run -a
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System-specific variables (absolute paths, private keys) should not be
versioned. Create a `.env` file (excluded from git) based on `.env.example`.

Hydra can reference env variables in configs:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Keep local configs out of version control</b></summary>

User/machine-specific settings go in `configs/local/default.yaml`, which is
loaded automatically but not tracked by git.

</details>

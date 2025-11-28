# Test scripts

Forward-pass and training sanity checks for the Lorentz attention module and full classifier.  
All tests rely on Python logging; attention internals can be inspected via `--attn-debug` and high log levels.

## Unified forward-pass tests

`test_forwards.py` validates tensor shapes, manifold constraints, and attention behaviour.

### Key flags
| Flag | Description |
|------|-------------|
| `--target {attn,class,both}` | Test attention only, classifier only, or both. |
| `--compute-scores {lorentz_inner,signed_dist}` | Attention scoring rule. |
| `--concat-operation {direct,log-radius}` | Head concatenation strategy. |
| `--value-agg riemannian` | Value aggregation (current default). |
| `--attn-debug` | Enables detailed debug logs from inside the attention block. |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Controls logging verbosity. |

### Usage
Forward-pass test for attention:
```bash
python ./tests/test_forwards.py --target attn
```

Attention debug mode:
```bash
python ./tests/test_forwards.py --target attn --attn-debug --log-level DEBUG
```

Classifier forward-pass:
```bash
python ./tests/test_forwards.py --target class
```

## Training/debug script

`test_train.py` provides a short training loop with logging, NaN tracing, and model checks.

### Key flags

| Flag                                     | Description                                                   |
| ---------------------------------------- | ------------------------------------------------------------- |
| `--model {personal,hypformer}`           | Selects which model to test.                                  |
| `--vocab-size INT`                       | Vocabulary size for synthetic data.                           |
| `--pad-id INT`                           | Padding token ID.                                             |
| `--embed-dim INT`                        | Embedding dimension.                                          |
| `--num-classes INT`                      | Number of output classes.                                     |
| `--num-heads INT`                        | Attention heads.                                              |
| `--curvature-k FLOAT`                    | Curvature parameter.                                          |
| `--seq-len INT`                          | Sequence length.                                              |
| `--batch-size INT`                       | Batch size.                                                   |
| `--num-steps INT`                        | Number of training steps.                                     |
| `--lr FLOAT`                             | Learning rate.                                                |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Controls logging verbosity.                                   |
| `--log-interval INT`                     | Steps between training log prints.                            |
| `--trace-nans`                           | Enables detailed parameter/gradient NaN/Inf checks each step. |

### Usage

Run (from root folder) with defaults:

```bash
python ./tests/test_train.py
```

Enable NaN tracing and verbose output:

```bash
python ./tests/test_train.py --trace-nans --log-level DEBUG
```


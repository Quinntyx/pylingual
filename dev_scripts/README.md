# Model Training

PyLingual's accuracy is dependent on having accurate segmentation and statement models [^1]. The segmentation model divides a list of bytecode instructions into groups for each source instruction. The statement model transforms each group of instructions into source code. The instructions for training these models is as follows:

## Dataset generation

Create a dataset JSON file based off the sample (`sample_jsons/py36-sample-data.json`).

The dataset directory should be structured like so, with only one `.py` file per directory:

```
dataset
├── 0
│   └── file.py
├── 1
│   └── file.py
...
├── 999
│   └── file.py
└── 1000
    └── file.py
```

The names of the inner directories and files do not matter. Then create the dataset:

```
uv run prepare_dataset.py <path to JSON>
```

## Segmentation model

Create a segmentation model JSON file based off the sample (`sample_jsons/py36-sample-segmentation.json`). Then train the model:

```
uv run train_models.py --segmentation <path to JSON>
```

## Statement model

Create a statement model JSON file based off the sample (`sample_jsons/py36-sample-statement.json`). Then train the model:

```
uv run train_models.py --statement <path to JSON>
```

Once models are trained, update `../pylingual/decompiler_config.yaml` or create a separate config file by replacing the old models with the newly trained ones.

[^1]: [pylingual models](https://huggingface.co/syssec-utd).

## Control flow model

The optional control flow model predicts how the translated statements of a code object nest, as a skeleton such as `<s0> { <s1> } else { <s2> }`. Its training data comes from synthetic programs plus a normal dataset JSON (`sample_jsons/py312-sample-control-flow-data.json`), and it is trained from a model JSON (`sample_jsons/py312-sample-control-flow.json`):

```
uv run -m dataset_generation.synthetic_source -o ./cache-dir/synth312 -n 20000 -v 3.12
uv run prepare_dataset.py sample_jsons/py312-sample-control-flow-data.json --stage generate --stage compile --stage cflow
uv run train_models.py --control-flow sample_jsons/py312-sample-control-flow.json --nproc_per_node 4
```

To use it, add `CONTROL_FLOW_MODEL: {REPO: <HuggingFace repo or absolute path>, REVISION: main}` to the version in `decompiler_config.yaml`. Code objects that fail the equivalence check fall back to the model's other beams, then to the control flow templates, then to re-segmentation. Changing the output of `pylingual/control_flow_reconstruction/model_view.py` needs new models.

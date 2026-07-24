# Pretrained model package contract

Registered pretrained models must use a predictable archive layout. The archive
name, top-level directory, and configuration stem must all match the
`MODEL_REGISTRY` key:

```text
<model_name>.zip
└── <model_name>/
    ├── <model_name>.yaml
    └── checkpoints/
        └── best.pdparams
```

`best.pdparams` must contain the model state dictionary directly:

```python
paddle.save(model.state_dict(), "best.pdparams")
```

Do not package a trainer checkpoint such as `{"model": state_dict, "step": ...}`
as `best.pdparams`. Training metadata belongs in a separate file. A model that
needs auxiliary pretrained components may add clearly named files under
`checkpoints/`, while its primary weight remains `best.pdparams`.

The package may include inference-only examples or assets beside
`checkpoints/`. Paths in the YAML must be relative to the package directory,
and package consumers must resolve them explicitly without changing the process
working directory.

Before publishing a registry entry:

1. Download and extract the final archive through
   `ppmat.utils.download.get_weights_path_from_url`.
2. Resolve it with `resolve_model_package_dir`.
3. Build the model from its packaged YAML.
4. Load `checkpoints/best.pdparams` without missing or unexpected parameters.
5. Run the documented inference command from a clean cache.
6. Record the final archive SHA256 in the release or pull-request validation
   notes.

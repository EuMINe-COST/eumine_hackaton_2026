# MatFed API Template (EuMINe DataBridge Hackathon 2026)

Template repository for hackathon participants.

## Repository layout

```
matfed-api-template/
├── README.md
├── requirements.txt
├── matfed_api/
│   ├── __init__.py
│   ├── predictor.py
│   └── schema.py
├── example_implementation/
│   ├── my_predictor.py
│   └── run_example.py
└── tests/
    ├── conftest.py
    ├── test_interface.py
    └── sample_structures/
```

## Quick start

```bash
pip install -r requirements.txt
pytest tests/test_interface.py -v
```

By default tests use `example_implementation.my_predictor.RandomForestPredictor`.

To test your own predictor:

```bash
export MY_PREDICTOR=mypkg.my_module.MyPredictor
pytest tests/test_interface.py -v
```

Required interface (`MatFedPredictor`):

- `load_model(self, model_path: str) -> None`
- `predict(self, structures: List[Structure]) -> List[Dict]`
- `describe(self) -> Dict`

Each prediction item must include:

- `formation_energy_per_atom` (float)
- `band_gap` (float)
- `model_id` (str)
- `data_sources_used` (list[str])

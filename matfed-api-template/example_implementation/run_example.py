from pathlib import Path

from pymatgen.core import Structure

from example_implementation.my_predictor import RandomForestPredictor


def main() -> None:
    sample_dir = Path("tests/sample_structures")
    structures = [
        Structure.from_file(path)
        for path in sorted(sample_dir.glob("*.cif"))
    ]

    predictor = RandomForestPredictor()
    predictor.load_model(".")

    print("Model description:")
    print(predictor.describe())
    print()

    predictions = predictor.predict(structures)
    for index, pred in enumerate(predictions, start=1):
        print(f"{index:02d}: {pred}")


if __name__ == "__main__":
    main()

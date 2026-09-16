from pathlib import Path

from semantic.data_loader import JIGSAWSKinematicsDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    print("=" * 60)
    print("JIGSAWS DATASET LOADER TEST")
    print("=" * 60)

    dataset = JIGSAWSKinematicsDataset(
        root_dir=DATA_DIR,
        sequence_length=32,
        stride=16,
        normalize=True
    )

    print(f"Dataset directory : {DATA_DIR}")
    print(f"Sequences found   : {len(dataset)}")
    print(f"Feature dimension : {dataset.feature_dim}")
    print(f"Sequence shape    : {dataset[0].shape}")
    print(f"First sample      :\n{dataset[0][:2]}")

    print("=" * 60)
    print("LOADER TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()

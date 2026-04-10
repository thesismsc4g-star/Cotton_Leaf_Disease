import argparse
from pathlib import Path
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gdown

from config import DATA_DIR

DEFAULT_FILE_ID = "1BENN1xbHgF1nPUI_a8JYMy5exfGOHcM7"


def download_and_extract(file_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "Cotton_Augmented_Dataset.zip"
    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, str(zip_path), quiet=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    dataset_dir = output_dir / "Cotton_Augmented_Dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError("Expected Cotton_Augmented_Dataset folder not found after unzip.")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    dataset_dir = download_and_extract(args.file_id, Path(args.output_dir))
    print(f"Dataset ready at: {dataset_dir}")


if __name__ == "__main__":
    main()

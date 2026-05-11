from pathlib import Path
import shutil

from backports.zstd import train_dict

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PACE_DIR = DATA_DIR / "raw" / "PACE2017B"
TRAIN_DIR = DATA_DIR / "train"

OUTPUT_DIR = DATA_DIR / "test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_files = {file.name for file in PACE_DIR.iterdir() if file.suffix == ".graph"}
train_files = {file.name for file in TRAIN_DIR.iterdir() if file.suffix == ".graph"}

duplicated_files = test_files & train_files
files_to_copy = test_files - duplicated_files

for file in files_to_copy:
    shutil.copy(PACE_DIR / file, OUTPUT_DIR)
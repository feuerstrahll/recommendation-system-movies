from pathlib import Path
import shutil
import zipfile

import kagglehub


def download_dataset() -> Path:
    """
    Загружает датасет The Movies Dataset с Kaggle.
    """
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
    print("Path to dataset files:", path)
    return Path(path)


def copy_csv_to_raw(dataset_path: Path):
    """
    Копирует все CSV-файлы из скачанного датасета в data/raw.
    Если внутри есть ZIP-архивы, распаковывает их и забирает CSV.
    """
    project_dir = Path(__file__).resolve().parent
    raw_dir = project_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Копируем обычные CSV
    for csv_file in dataset_path.rglob("*.csv"):
        destination = raw_dir / csv_file.name
        shutil.copy2(csv_file, destination)
        print(f"Copied: {csv_file.name}")

    # Если вдруг есть ZIP-архивы — распаковываем CSV из них
    for zip_file in dataset_path.rglob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as archive:
            for file_name in archive.namelist():
                if file_name.endswith(".csv"):
                    destination = raw_dir / Path(file_name).name

                    with archive.open(file_name) as source:
                        with open(destination, "wb") as target:
                            shutil.copyfileobj(source, target)

                    print(f"Extracted: {Path(file_name).name}")

    print(f"\nAll CSV files are saved in: {raw_dir}")


if __name__ == "__main__":
    dataset_path = download_dataset()
    copy_csv_to_raw(dataset_path)
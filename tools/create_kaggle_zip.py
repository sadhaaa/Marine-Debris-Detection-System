from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "DL_project_kaggle.zip"

EXCLUDED_DIRECTORIES = {
    ".venv",
    ".git",
    "__pycache__",
}

EXCLUDED_FILES = {
    "DL_project_cloud.zip",
    "DL_project_kaggle.zip",
}


def main() -> None:
    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(PROJECT_ROOT)

        if any(
            part in EXCLUDED_DIRECTORIES
            for part in relative_path.parts
        ):
            continue

        if path.name in EXCLUDED_FILES:
            continue

        files.append(path)

    print(f"Files to archive: {len(files)}")

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:

        for index, path in enumerate(files, start=1):
            relative_path = path.relative_to(PROJECT_ROOT)

            # Convert Windows "\" paths to standard "/" paths.
            archive_name = relative_path.as_posix()

            archive.write(
                path,
                arcname=archive_name,
            )

            if index % 500 == 0:
                print(
                    f"Archived {index}/{len(files)} files..."
                )

    print()
    print("=" * 60)
    print("ZIP CREATION COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_ZIP}")
    print(
        f"Size: "
        f"{OUTPUT_ZIP.stat().st_size / (1024 * 1024):.1f} MB"
    )


if __name__ == "__main__":
    main()
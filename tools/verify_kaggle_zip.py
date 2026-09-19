from pathlib import Path
import zipfile


zip_path = Path("..") / "DL_project_kaggle.zip"

with zipfile.ZipFile(zip_path, "r") as archive:
    names = archive.namelist()

bad_paths = [
    name
    for name in names
    if "\\" in name
]

print("=" * 60)
print("KAGGLE ZIP VERIFICATION")
print("=" * 60)

print(f"ZIP: {zip_path.resolve()}")
print(f"Total files: {len(names)}")
print(f"Backslash paths: {len(bad_paths)}")

print("\nFirst 15 paths:")
for name in names[:15]:
    print(name)

if bad_paths:
    print("\nERROR: Backslash paths detected!")
    print("\nExamples:")
    for name in bad_paths[:10]:
        print(name)
else:
    print("\nPASS: All archive paths use '/'.")

print("=" * 60)
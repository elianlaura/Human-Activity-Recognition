import os
import shutil


def save_folder(src, dest="dir"):
    """
    Copy a folder and its subfolders/files to 'dest',
    excluding certain file extensions, filenames, and folders.
    """

    # File extensions to exclude
    exclude_ext = {".out", ".err", ".sh", ".npy", ".gitignore", ".png", ".jpg", ".mat"}

    # Filenames to exclude
    exclude_files = {"pip-reqs.txt", "README.md", "PaD-TS.yml", "init.py"}

    # Folder names to exclude
    exclude_dirs = {
        ".vscode",
        ".venv",
        "fmri",
        "figs",
        "OUTPUT",
        "results",
        "dataset",
        "__pycache__",
    }

    # Ensure destination exists
    os.makedirs(dest, exist_ok=True)

    # Always keep submit scripts together under a dedicated folder.
    submit_dest = os.path.join(dest, "submit")
    os.makedirs(submit_dest, exist_ok=True)

    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        if name.startswith("submit_") and os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(submit_dest, name))

    for root, dirs, files in os.walk(src):

        # Remove excluded directories from traversal
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Compute relative path
        rel_path = os.path.relpath(root, src)
        dest_path = os.path.join(dest, rel_path)

        os.makedirs(dest_path, exist_ok=True)

        for file in files:
            # Skip excluded files
            if file in exclude_files:
                continue
            if os.path.splitext(file)[1] in exclude_ext:
                continue

            # Copy file
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_path, file)

            # Avoid machine-specific links (e.g. venv binaries) and broken links.
            if os.path.islink(src_file):
                continue

            shutil.copy2(src_file, dst_file)

def remove_pycache(dest):
    """ Remove __pycache__ directories from the destination folder.
    """
    # --- Extra step: remove __pycache__ inside "code" folder if it exists ---
    pycache_path = os.path.join(dest, "__pycache__")
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)
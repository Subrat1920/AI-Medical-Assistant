import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format = '[%(asctime)s]: %(messages)s:'
)

lists_of_files = [
    "src/___init__.py",
    "src/helper.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]

for filepath in lists_of_files:
    filepath = Path(filepath)
    filedir, filename  = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f'Creating Empty file : {filepath}')

    else:
        logging.info(f"{filename} is already exists")
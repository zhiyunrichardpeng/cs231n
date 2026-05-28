import json

with open('Dropout.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replace cell 0 (Colab setup) with local setup
new_setup = """# Local VS Code Setup (replaces Google Colab mount)
# This cell sets up the environment to run locally

import os
import sys

# Get the directory where this notebook is located
NOTEBOOK_DIR = os.getcwd()

# Add the assignment directory to Python path if not already there
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

print(f"Working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

# Verify the cs231n module is accessible
assert os.path.exists("cs231n"), "cs231n folder not found. Make sure you are in the assignment2 directory."
print("Setup complete! cs231n module is accessible.")"""

nb['cells'][0]['source'] = [new_setup]

with open('Dropout.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Replaced Colab setup cell with local VS Code setup.")

#!/usr/bin/env python3
"""
Script to fix NumPy 2.0 compatibility issues with NeMo.
The np.sctypes dictionary was removed in NumPy 2.0.
This script adds a compatibility patch.
"""
import os
import re

# Path to the segment.py file in NeMo
venv_path = "nemo-asr-venv"
segment_path = os.path.join(
    venv_path, 
    "lib/python3.12/site-packages/nemo/collections/asr/parts/preprocessing/segment.py"
)

# Check if file exists
if not os.path.exists(segment_path):
    raise FileNotFoundError(f"Could not find {segment_path}")

# Read the file
with open(segment_path, 'r') as f:
    content = f.read()

# Add np.sctypes compatibility
compatibility_patch = """
# NumPy 2.0 compatibility patch
try:
    np.sctypes
except AttributeError:
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, np.bytes_, np.str_, np.void]
    }
"""

# Check if import numpy as np is in the file
if 'import numpy as np' in content:
    # Insert the compatibility patch after the import
    pattern = r'import numpy as np'
    content = re.sub(pattern, f'import numpy as np\n{compatibility_patch}', content)

    # Write the patched file
    with open(segment_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {segment_path} for NumPy 2.0 compatibility")
else:
    print(f"Could not find 'import numpy as np' in {segment_path}")

print("Done!") 
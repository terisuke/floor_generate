#!/usr/bin/env python3

"""
Wrapper script to run inference_test.py with the compatibility patches applied.
"""

import os
import sys
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import patch_diffusers
patch_diffusers.apply_patches()

inference_script = os.path.join(current_dir, "inference_test.py")
site_mask = os.path.join(parent_dir, "test_output", "test_site_mask.png")
output_dir = os.path.join(parent_dir, "test_output")

cmd = [
    sys.executable,
    inference_script,
    "--site_mask", site_mask,
    "--output_dir", output_dir,
    "--steps", "5"  # Use fewer steps for faster testing
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd)
sys.exit(result.returncode)

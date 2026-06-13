# Create a file named patch_advertorch.py
import os
import re

# Path to the problematic file
file_path = '/.local/lib/python3.9/site-packages/advertorch/attacks/fast_adaptive_boundary.py'

# Read the file
with open(file_path, 'r') as f:
    content = f.read()

# Replace the import
new_content = content.replace(
    'from torch.autograd.gradcheck import zero_gradients',
    '''
# Define zero_gradients function as it was removed from PyTorch
def zero_gradients(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()
'''
)

# Write the file back
with open(file_path, 'w') as f:
    f.write(new_content)

print("advertorch patched successfully!")
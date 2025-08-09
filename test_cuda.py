import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    print("✅ SUCCESS: Using GPU!")
else:
    print("❌ Using CPU")
    device = torch.device("cpu")
print("Using device:", device)
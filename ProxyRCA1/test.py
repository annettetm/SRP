import sys
import torch

print("Python version:")
print(sys.version)

print("\nPyTorch version:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# Create a tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)

print("\nInput tensor:")
print(x)

# Simple PyTorch calculation
y = x * 2 + 1

print("\nResult:")
print(y)

# Small neural network
model = torch.nn.Linear(2, 1).to(device)

output = model(x)

print("\nNeural network output:")
print(output)

print("\nPyTorch test completed successfully!")
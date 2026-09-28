from datasets import load_dataset

print("Downloading dataset...")

dataset = load_dataset("sfarrukhm/intel-image-classification")

print("Dataset downloaded successfully!")
print(dataset)
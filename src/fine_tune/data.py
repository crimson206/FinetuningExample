import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from typing import Dict, Tuple
import tarfile

class Caltech101Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
    
        self.classes = os.listdir(directory)
        self.classes.sort()
        self.index_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.samples = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read all the files and folders in the directory
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith('jpg'):
                    class_name = os.path.basename(root)  # This should give you just the class name like 'watch'
                    self.samples.append((os.path.join(root, filename), class_name))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.classes.index(class_name)
        
        if self.transform:
            image = self.transform(image)

        return image, label

class TransformsHolder:
    transform_default = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
def create_category_mapping(directory: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Generates two dictionaries mapping category names to indices and vice versa.
    
    Parameters:
    directory (str): The dataset directory path.
    
    Returns:
    Tuple[Dict[int, str], Dict[str, int]]: index_to_category, category_to_index
    """

    categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    categories.sort()  # Sort the categories alphabetically

    index_to_category: Dict[int, str] = {i: category for i, category in enumerate(categories)}
    category_to_index: Dict[str, int] = {category: i for i, category in enumerate(categories)}

    return index_to_category, category_to_index

def extract_tar(tar_path, extract_path="caltech101"):
    # Check if the extraction path already exists
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    else:
        print(f"Directory '{extract_path}' already exists. Extraction skipped.")
from torch.utils.data import Dataset
import pandas as pd

class EnglishToSpanish(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx+1]
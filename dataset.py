from torch.utils.data import Dataset
import pandas as pd

class EnglishToSpanish(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        res = self.df.iloc[idx]
        return res["english"], res["spanish"]
from torch.utils.data import Dataset
import pandas as pd

class EnglishToSpanish(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df[idx+1]
    
    def preprocess(self, tokenizer, max_len, special_tokens):
        """
        Adds padding and beginning/end of sequence tokens.
        """
        self.df = self.df.map(lambda x: self.__process_row(x, tokenizer, max_len, special_tokens))

    def __process_row(self, x, tokenizer, max_len, special_tokens):
        tok_len = len(tokenizer.encode(x)) + 2 # Add 2 because we add two speciall tokens below before padding
        x = special_tokens["bos"] + x + special_tokens["eos"]
        for _ in range(max_len - tok_len):
            x += special_tokens["pad"]
        return x
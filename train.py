# Script to actually train the model
# import kagglehub
from torch.utils.data import random_split, DataLoader

from dataset import EnglishToSpanish
from model import Transformer

def main():
    # TODO: Figure out how to do this, but download to data folder instead
    # path = kagglehub.dataset_download("lonnieqin/englishspanish-translation-dataset")
    # print("Path to dataset files:", path)

    data = EnglishToSpanish("./data/data.csv")

    train_data, test_data = random_split(data, [0.7, 0.3])
    
    train_loader = DataLoader(train_data)
    test_loader = DataLoader(test_data)



if __name__ == "__main__":
    main()
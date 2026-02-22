import pandas as pd
from datasets import load_dataset

#Hyperparams to tune
BATCH_SIZE = 9000

def main():
    dataset_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=f"train[:{BATCH_SIZE}]")
    dataset_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    df_artificial = dataset_artificial.to_pandas()
    print(df_artificial.shape)
    print(df_artificial.columns)

    df_labeled = dataset_labeled.to_pandas()
    print(df_labeled.shape)
    print(df_labeled.columns)

    #Matches data in the original dataset.
    print(df_artificial.head())
    print(df_artificial.tail())
    print(df_labeled.head())
    print(df_labeled.tail())

    # Optional - Save as CSV for easier loading 
    #df_artificial.to_csv("medqa_data_artificial.csv", index=False)
    #df_labeled.to_csv("medqa_data_labeled.csv", index=False)
    #print("Saved")


if __name__ == "__main__":
    main()

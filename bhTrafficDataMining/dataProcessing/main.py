import os
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
import polars as pl


if __name__ == "__main__":
    load_dotenv("./config/.env")
    json_files = glob("./data/jan/*.json")
    day_folders = os.listdir("./data/jan/")

    dataframes_list = []
    for folder in tqdm(day_folders):
        json_files = glob(f"./data/jan/{folder}/*.json")
        for file in tqdm(json_files):
            df = pl.read_json(file)
            filtered_df = (
                df.lazy()
                .filter(pl.col("ENDEREÇO").str.contains(f"{os.environ["TARGET"]}"))
                .collect()
            )
            dataframes_list.append(filtered_df)

    month_df = pl.concat(dataframes_list)
    print(month_df)

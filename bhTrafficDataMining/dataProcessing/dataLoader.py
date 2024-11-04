import os
from glob import glob
from dataclasses import dataclass
from tqdm import tqdm
import polars as pl


@dataclass
class DataLoader:
    dir: str
    target: str

    def list_files(self):
        months_folders = os.listdir(self.dir)
        json_files_list = []

        for m_folder in months_folders:
            for d_folder in m_folder:
                json_files = glob(f"{self.dir}/{m_folder}/{d_folder}/*.json")
                json_files_list.extend(json_files)

        return json_files_list

    def load_data(self):
        json_files_list = self.list_files()
        dataframes_list = []
        for file in tqdm(json_files_list):
            df = pl.read_json(file)
            filtered_df = (
                df.lazy()
                .filter(pl.col("ENDEREÇO").str.contains(f"{self.target}"))
                .collect()
            )
            dataframes_list.append(filtered_df)

        month_df = pl.concat(dataframes_list)
        print(month_df)

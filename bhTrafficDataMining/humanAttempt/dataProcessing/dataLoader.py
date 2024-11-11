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
            d_folder_list = os.listdir(f"{self.dir}/{m_folder}")
            for d_folder in d_folder_list:
                json_files = glob(f"{self.dir}/{m_folder}/{d_folder}/*.json")
                json_files_list.extend(json_files)

        return json_files_list

    def load_data(self) -> pl.DataFrame:
        json_files_list = self.list_files()
        dataframes_list = []
        for file in tqdm(json_files_list):
            df = pl.read_json(file)
            filtered_df = (
                df.lazy()
                .filter(pl.col("ENDEREÇO").str.contains(f"{self.target}"))
                .collect()
            )
            filtered_df = filtered_df.with_columns(
                (pl.col("VELOCIDADE AFERIDA") > pl.col("VELOCIDADE DA VIA")).alias(
                    "ULTRAPASSOU LIMITE"
                )
            )
            dataframes_list.append(filtered_df)

        months_df = pl.concat(dataframes_list)

        return months_df


if __name__ == "__main__":
    data_loader = DataLoader("./bhTrafficDataMining/dataProcessing/data/", "Contorno")
    data_loader.load_data()

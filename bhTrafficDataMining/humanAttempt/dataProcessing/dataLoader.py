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
            m_folder = os.path.join(self.dir, m_folder)
            if os.path.isdir(m_folder):
                d_folder_list = os.listdir(m_folder)
                for d_folder in d_folder_list:
                    d_folder = os.path.join(m_folder, d_folder)
                    json_files = glob(f"{d_folder}/*.json")
                    json_files_list.extend(json_files)

        return json_files_list

    def filter_location(self, p_df: pl.DataFrame):
        filtered_df = (
            p_df.lazy()
            .filter(pl.col("ENDEREÇO").str.contains(f"{self.target}"))
            .collect()
        )

        return filtered_df

    def check_above_speed_limite(self, p_df: pl.DataFrame):
        df = p_df.with_columns(
            pl.col("VELOCIDADE AFERIDA").cast(pl.Float32),
            pl.col("VELOCIDADE DA VIA").cast(pl.Float32),
        )
        df = df.with_columns(
            (pl.col("VELOCIDADE AFERIDA") > pl.col("VELOCIDADE DA VIA")).alias(
                "ULTRAPASSOU LIMITE"
            )
        )

        return df

    def categorize_time_of_day_extended(self, hour):
        if 0 <= hour < 6:
            return "Madrugada"
        elif 6 <= hour < 9 or 17 <= hour < 20:
            return "Pico"
        elif 9 <= hour < 12:
            return "Manhã"
        elif 12 <= hour < 17:
            return "Tarde"
        else:
            return "Noite"

    def discretize_datetime(self, p_df: pl.DataFrame):
        df = p_df.with_columns(
            pl.col("DATA HORA").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S")
        )

        df = (
            df.lazy()
            .with_columns(
                pl.col("DATA HORA")
                .dt.hour()
                .map_elements(
                    self.categorize_time_of_day_extended, return_dtype=pl.String
                )
                .alias("PERIODO DIA")
            )
            .collect()
        )

        return df

    def load_data(self) -> pl.DataFrame:
        json_files_list = self.list_files()
        dataframes_list = []

        for file in tqdm(json_files_list):
            df = pl.read_json(file)
            filtered_df = self.filter_location(df)
            filtered_df = self.check_above_speed_limite(filtered_df)

            filtered_df = self.discretize_datetime(filtered_df)

            filtered_df = filtered_df[
                "PERIODO DIA",
                "ENDEREÇO",
                "SENTIDO",
                "FAIXA",
                "CLASSIFICAÇÃO",
                "TAMANHO",
                "VELOCIDADE DA VIA",
                "VELOCIDADE AFERIDA",
                "ULTRAPASSOU LIMITE",
            ]
            dataframes_list.append(filtered_df)

        months_df = pl.concat(dataframes_list)

        return months_df


if __name__ == "__main__":
    data_loader = DataLoader("./bhTrafficDataMining/dataProcessing/data/", "Contorno")
    data_loader.load_data()

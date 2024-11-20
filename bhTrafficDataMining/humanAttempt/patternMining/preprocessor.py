from glob import glob
from dataclasses import dataclass
from tqdm import tqdm
import polars as pl


@dataclass
class DataPreprocessor:
    dir: str

    def load_database(self, p_percentage) -> pl.DataFrame:
        csv_files_list = glob(f"{self.dir}/*.csv")
        percent_index = int(round(len(csv_files_list) * p_percentage))
        dataframes_list = []
        for file in tqdm(csv_files_list[:percent_index]):
            df = pl.read_csv(file)
            df = self.discretize_speed(df)
            dataframes_list.append(df)

        return pl.concat(dataframes_list)

    def get_speep_intervals(self, p_speed):
        if 0 <= p_speed < 50:
            return "velocidadeModerada"
        elif 50 <= p_speed < 100:
            return "velocidadeAlta"
        elif p_speed >= 100:
            return "velocidadeAltissima"

    def discretize_speed(self, p_df: pl.DataFrame):
        df = (
            p_df.lazy()
            .with_columns(
                pl.col("VELOCIDADE AFERIDA")
                .map_elements(self.get_speep_intervals, return_dtype=pl.String)
                .alias("VELCOIDADE")
            )
            .collect()
        )

        return df

    def drop_undesired_columns(
        self, p_df: pl.DataFrame, p_undesired_columns: list
    ) -> pl.DataFrame:
        return p_df.drop(p_undesired_columns)

    def get_preprocessed_database(self, p_percentage: float, p_undesired_columns: list):
        df = self.load_database(p_percentage)
        df = self.drop_undesired_columns(df, p_undesired_columns)
        df = df.to_dummies()
        df = df.cast(pl.Boolean)

        return df


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(
        "bhTrafficDataMining/humanAttempt/dataProcessing/dataProcessed"
    )

    df = data_preprocessor.get_preprocessed_database(
        0.5, p_undesired_columns=["VELOCIDADE AFERIDA", "VELOCIDADE DA VIA", "TAMANHO"]
    )
    print(df)

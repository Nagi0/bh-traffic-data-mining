import polars as pl
from tqdm import tqdm


if __name__ == "__main__":
    df = pl.read_csv(
        "bhTrafficDataMining/dataProcessing/data/FEVEREIRO_2022/FEVEREIRO_2022.csv",
        separator=";",
    )
    print(df)
    # Selecionando as colunas relevantes para a análise transacional
    transactional_data_polars = df.select(
        [
            "VELOCIDADE AFERIDA",
            "FAIXA",
            "LATITUDE",
            "LONGITUDE",
            "ULTRAPASSOU LIMITE",
        ]
    )

    # Convertendo os dados para uma lista de dicionários
    transactions_polars = transactional_data_polars.to_dicts()

    # Convertendo a lista de dicionários para uma lista de listas representando as transações
    transactions = [
        [str(value) for value in transaction.values() if value is not None]
        for transaction in transactions_polars
    ]

    # Exibindo as transações
    for transaction in transactions:
        print(transaction)

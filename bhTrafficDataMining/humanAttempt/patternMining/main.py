import polars as pl
from mlxtend.frequent_patterns import fpgrowth, association_rules
from preprocessor import DataPreprocessor


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(
        "bhTrafficDataMining/humanAttempt/dataProcessing/dataProcessed"
    )

    df = data_preprocessor.get_preprocessed_database(
        1.0, p_undesired_columns=["VELOCIDADE AFERIDA", "VELOCIDADE DA VIA", "TAMANHO"]
    )
    df = df.lazy().filter(pl.col("ULTRAPASSOU LIMITE_true") == True).collect()
    df = df.drop(["ULTRAPASSOU LIMITE_true", "ULTRAPASSOU LIMITE_false"])
    df = df.to_pandas()
    print(df)

    frequent_itemsets = fpgrowth(df, min_support=0.25, use_colnames=True, verbose=1)
    print(frequent_itemsets)
    frequent_itemsets.to_csv("bhTrafficDataMining/humanAttempt/resultados.csv")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.75,
        num_itemsets=df.shape[0],
    )
    print(rules)
    rules.to_csv("bhTrafficDataMining/humanAttempt/association_rules_conf75.csv")

import os
from dotenv import load_dotenv
from dataLoader import DataLoader


if __name__ == "__main__":
    load_dotenv("bhTrafficDataMining/humanAttempt/dataProcessing/config/.env")
    data_loader = DataLoader("bhTrafficDataMining/data", os.environ["TARGET"])
    df = data_loader.load_data()
    df.write_csv(
        "bhTrafficDataMining/humanAttempt/dataProcessing/dataProcessed/ABRIL_2022.csv",
    )
    print(df["ENDEREÇO"].value_counts())

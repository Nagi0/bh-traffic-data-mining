import os
from dotenv import load_dotenv
from dataLoader import DataLoader


if __name__ == "__main__":
    load_dotenv("bhTrafficDataMining/dataProcessing/config/.env")
    data_loader = DataLoader(
        "bhTrafficDataMining/dataProcessing/data/", os.environ["TARGET"]
    )
    df = data_loader.load_data()
    print(df)

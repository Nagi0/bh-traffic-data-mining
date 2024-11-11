import os
import polars as pl
from tqdm import tqdm

# Path to the database (root directory containing month folders)
root_directory = "bhTrafficDataMining/data"

# Output directory for CSV files
output_directory = "bhTrafficDataMining/llmAttempt/dataProcessed"

# Create output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through each month folder in the root directory, using tqdm to show progress
for month_folder in tqdm(os.listdir(root_directory), desc="Processing months"):
    month_path = os.path.join(root_directory, month_folder)

    # Check if the current path is a directory (i.e., a month folder)
    if os.path.isdir(month_path):
        # List to collect dataframes for all JSON files in the current month
        monthly_dataframes = []

        # Iterate through each day folder inside the current month folder, using tqdm to show progress
        for day_folder in tqdm(
            os.listdir(month_path),
            desc=f"Processing days in {month_folder}",
            leave=False,
        ):
            day_path = os.path.join(month_path, day_folder)

            # Check if the current path is a directory (i.e., a day folder)
            if os.path.isdir(day_path):
                # Iterate through each JSON file in the current day folder
                for json_file in os.listdir(day_path):
                    # Process only files with a .json extension
                    if json_file.endswith(".json"):
                        json_path = os.path.join(day_path, json_file)

                        # Read JSON file with polars and append it to the list of dataframes
                        try:
                            df = pl.read_json(json_path)

                            # Ensure required columns exist before processing
                            required_columns = [
                                "ENDEREÇO",
                                "VELOCIDADE AFERIDA",
                                "VELOCIDADE DA VIA",
                            ]
                            if all(col in df.columns for col in required_columns):
                                # Filter dataframe to include only rows where 'ENDEREÇO' contains 'Contorno'
                                filtered_df = df.filter(
                                    df["ENDEREÇO"].str.contains("Contorno")
                                )

                                # Drop columns that are not needed
                                filtered_df = filtered_df.drop(
                                    [
                                        "ID DE ENDEREÇO",
                                        "NUMERO DE SÉRIE",
                                        "LATITUDE",
                                        "LONGITUDE",
                                        "MILESEGUNDO",
                                        "ID EQP",
                                    ]
                                )

                                # Add a boolean column indicating if 'VELOCIDADE AFERIDA' is above 'VELOCIDADE DA VIA'
                                filtered_df = filtered_df.with_columns(
                                    (
                                        filtered_df["VELOCIDADE AFERIDA"]
                                        > filtered_df["VELOCIDADE DA VIA"]
                                    ).alias("ACIMA VELOCIDADE PERMITIDA")
                                )

                                monthly_dataframes.append(filtered_df)
                            else:
                                print(
                                    f"Skipping {json_path} due to missing required columns."
                                )
                        except Exception as e:
                            # Print an error message if there is an issue reading the JSON file
                            print(f"Error reading {json_path}: {e}")

        # Concatenate all dataframes for the current month if any data was collected
        if monthly_dataframes:
            # Concatenate all dataframes into a single dataframe for the month
            month_df = pl.concat(monthly_dataframes, how="vertical")

            # Save the concatenated dataframe to a CSV file named after the month folder
            output_csv_path = os.path.join(output_directory, f"{month_folder}.csv")
            month_df.write_csv(output_csv_path)
            print(f"Saved CSV for {month_folder} at {output_csv_path}")
        else:
            # Print a message if no data was found for the current month
            print(f"No data found for {month_folder}")

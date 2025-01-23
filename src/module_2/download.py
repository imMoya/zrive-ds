import os
from pathlib import Path

import boto3
from dotenv import load_dotenv


def download_files(
    s3: boto3.client, 
    bucket_name: str,
    folder_path: str,
    files: list[str],
    output_dir: str | Path,
) -> None:
    for file_name in files:
        file_key = os.path.join(folder_path, file_name)
        os.makedirs(output_dir, exist_ok=True)
        local_file_name = os.path.join(output_dir, file_name)
        try:
            print(f'Downloading {file_key}...')
            s3.download_file(bucket_name, file_key, local_file_name)
            print(f'File {file_name} downloaded successfully.')
        except Exception as e:
            print(f'Error downloading {file_name}: {e}')


if __name__ == "__main__":
    load_dotenv()

    # aws secrets, buckets, folders and files...
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    bucket_name = 'zrive-ds-data'
    folder_path_exe_1 = 'groceries/sampled-datasets/'
    folder_path_exe_2 = 'groceries/box_builder_dataset/'
    files_exe_1 = [
        'orders.parquet',
        'regulars.parquet',
        'abandoned_cart.parquet',
        'inventory.parquet',
        'users.parquet',
    ]
    files_exe_2 = [
        'sampled_box_builder_df.csv'
    ]
    output_dir = Path("datasets", "module_2")

    # connection to s3
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )
    
    # download files
    download_files(
        s3 = s3, 
        bucket_name = bucket_name,
        folder_path = folder_path_exe_1,
        files = files_exe_1,
        output_dir = output_dir,
    )
    download_files(
        s3 = s3, 
        bucket_name = bucket_name,
        folder_path = folder_path_exe_2,
        files = files_exe_2,
        output_dir = output_dir,
    )

    try:
        file_key = "groceries/box_builder_dataset/sampled_box_builder_df.csv"
        s3.head_object(Bucket=bucket_name, Key=file_key)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Error: {file_key} not found in the bucket {bucket_name}.")
        else:
            print(f"Unexpected error: {e}")
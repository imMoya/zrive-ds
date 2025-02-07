import os
import subprocess
from pathlib import Path

import boto3
from dotenv import load_dotenv


def list_files_in_folder(
    s3: boto3.client, bucket_name: str, folder_path: str
) -> list[str]:
    """
    List all files inside a specified folder in an S3 bucket.

    :param s3: Boto3 S3 client.
    :param bucket_name: Name of the S3 bucket.
    :param folder_path: Folder path inside the bucket (e.g., 'path/to/folder/').
    :return: List of file names in the specified folder.
    """
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
        if 'Contents' not in response:
            print(f'No files found in folder: {folder_path}')
            return []

        file_names = [
            obj['Key'].replace(folder_path, '', 1)
            for obj in response['Contents']
            if obj['Key'] != folder_path  # Exclude the folder itself if present
        ]
        return file_names
    except Exception as e:
        print(f'Error listing files in folder {folder_path}: {e}')
        return []


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


def download_files_cli(
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
            subprocess.run(
                [
                    'aws',
                    's3',
                    'cp',
                    f's3://{bucket_name}/{file_key}',
                    local_file_name,
                ],
                check=True,
            )
            print(f'File {file_name} downloaded successfully.')
        except subprocess.CalledProcessError as e:
            print(f'Error downloading {file_name}: {e}')


if __name__ == '__main__':
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
        'abandoned_carts.parquet',
        'inventory.parquet',
        'users.parquet',
    ]
    files_exe_2 = ['feature_frame.csv']
    output_dir = Path('datasets', 'module_2')

    # connection to s3
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # download files
    download_files_cli(
        bucket_name=bucket_name,
        folder_path=folder_path_exe_1,
        files=files_exe_1,
        output_dir=output_dir,
    )
    download_files_cli(
        bucket_name=bucket_name,
        folder_path=folder_path_exe_2,
        files=files_exe_2,
        output_dir=output_dir,
    )

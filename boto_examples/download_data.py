import os
import boto3
from dotenv import load_dotenv
from botocore.client import Config

if __name__ == "__main__":
    load_dotenv()

    # Set up AWS credentials
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Create an S3 client
    session = boto3.session.Session()
    s3_client = session.client(
        service_name="s3",
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version='s3v4'),
    )

    # Upload a local file to the specified S3 bucket
    s3_client.download_file(
        "ctrbuckettest",
        "/tests/sampled_preprocessed_train_50k.csv",
        "data/raw/sampled_preprocessed_train_50k.csv",
    )
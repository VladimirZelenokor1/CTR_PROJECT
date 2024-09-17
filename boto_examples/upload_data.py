import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv

if __name__ == '__main__':
    # Load environment variables from.env file
    load_dotenv()

    # Get AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Create a new S3 client using AWS credentials
    session = boto3.session.Session()
    s3_client = session.client(
        service_name="s3",
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version='s3v4'),
    )

    # Upload a local file to the specified S3 bucket
    s3_client.upload_file(
        "D:/Projects/CTR_PROJECT/tests/sampled_preprocessed_train_50k.csv",
        "ctrbuckettest",
        "/tests/sampled_preprocessed_train_50k.csv",
    )
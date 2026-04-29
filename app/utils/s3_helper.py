import os
import boto3
import io
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def upload_file_to_s3(file_bytes: bytes, filename: str) -> str:
    """
    Uploads a file byte stream to AWS S3 using boto3.
    Returns the S3 URI if successful.
    """
    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is missing.")


    s3_client = boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-1"))

    try:
        file_obj = io.BytesIO(file_bytes)
        
        # Upload to S3
        s3_client.upload_fileobj(file_obj, bucket_name, filename)
        
        s3_uri = f"s3://{bucket_name}/{filename}"
        return s3_uri
        
    except ClientError as e:
        logger.error(f"AWS Boto3 ClientError: {e}")
        raise Exception("Failed to upload file to S3.")
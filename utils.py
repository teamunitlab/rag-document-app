from fastapi import HTTPException, UploadFile, status
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import uuid
import boto3
import magic
import os
import re
from urllib.parse import quote, unquote
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File Upload Allowed Types and MIME Types
ALLOWED_EXTENSIONS = {"pdf", "tiff", "png", "jpeg", "jpg"}
ALLOWED_MIME_TYPES = {
    "pdf": "application/pdf",
    "tiff": "image/tiff",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
}

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

# Configure logging


def configure_logging():
    """
    Configure logging for the application.
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("app.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Validate file type and MIME type


async def validate_file(file: UploadFile):
    """
    Validate the file type and MIME type.
    """
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File type {file_extension} not allowed"
        )

    mime_type = magic.from_buffer(file.file.read(2048), mime=True)
    file.file.seek(0)  # Reset file pointer to the beginning
    if mime_type != ALLOWED_MIME_TYPES[file_extension]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MIME type for {file_extension}: {mime_type}",
        )


# Upload file to S3


async def upload_file(logger, file: UploadFile):
    """
    Upload the validated file to S3 and return the file URL.
    """
    try:
        # Validate the file
        await validate_file(file)

        # Generate a unique file identifier
        unique_id = str(uuid.uuid4())
        file_extension = file.filename.split(".")[-1].lower()
        file_key = f"{unique_id}.{file_extension}"
        metadata = {"filename": quote(file.filename)}

        # Upload file to S3
        s3_client.upload_fileobj(
            file.file, BUCKET_NAME, file_key, ExtraArgs={"Metadata": metadata}
        )
        file_url = (
            f"https://{BUCKET_NAME}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{file_key}"
        )
        logger.info(f"File {file.filename} uploaded successfully.")
        return unique_id, file_url

    except NoCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AWS S3 Credentials not available",
        )
    except PartialCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Incomplete AWS S3 credentials provided",
        )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "AccessDenied":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="AWS S3 Access denied. \
                Check your bucket policy and IAM permissions.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ClientError: {e.response['Error']['Message']}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Validate S3 URL


async def aws_s3_validate_url(url):
    """
    Validate the provided S3 URL.
    """
    s3_bucket_pattern = re.compile(
        r"^https://(?P<BUCKET_NAME>[a-zA-Z0-9\-]+)\.s3\.(?P<AWS_DEFAULT_REGION>[a-zA-Z0-9-]+)\.amazonaws\.com/?.*$"
    )
    match = s3_bucket_pattern.match(url)
    if not match:
        raise HTTPException(status_code=404, detail="URL not found")


# Retrieve filename from S3 object metadata


async def get_filename_s3(file_key):
    """
    Retrieve the filename from the S3 object metadata.
    """
    try:
        response = s3_client.head_object(Bucket=BUCKET_NAME, Key=file_key)
        return unquote(response["Metadata"]["filename"])
    except NoCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AWS S3 Credentials not available",
        )
    except PartialCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Incomplete AWS S3 credentials provided",
        )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "AccessDenied":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="AWS S3 Access denied. \
                Check your bucket policy and IAM permissions.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ClientError: {e.response['Error']['Message']}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

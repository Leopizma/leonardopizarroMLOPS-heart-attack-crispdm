import json
import os
from datetime import datetime
from typing import Dict, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# Cargar variables
load_dotenv()

#AWS
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")  #

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_s3_client():
    session_kwargs = {"region_name": AWS_REGION}

    # Si hay claves explícitas (desarrollo local), úsalas
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs.update(
            {
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            }
        )

    session = boto3.Session(**session_kwargs)
    return session.client("s3")


def save_prediction_to_s3(data: Dict[str, Any]) -> None:
   
    if not S3_BUCKET_NAME:
        print("S3_BUCKET_NAME no está definido. No se guardará en S3.")
        return

    s3 = get_s3_client()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"predictions/{timestamp}.json"

    body = json.dumps(data)

    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Predicción guardada en s3://{S3_BUCKET_NAME}/{key}")
    except (BotoCoreError, ClientError) as e:
        print(f"Error guardando en S3: {e}")

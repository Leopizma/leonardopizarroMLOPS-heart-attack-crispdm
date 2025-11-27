import json
import os
from datetime import datetime
import uuid
from typing import Dict, Any

import boto3
from dotenv import load_dotenv
load_dotenv()



AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


def get_s3_client():
    
    session_kwargs = {"region_name": AWS_REGION}

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if aws_access_key_id and aws_secret_access_key:
        session_kwargs.update(
            {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            }
        )

    session = boto3.Session(**session_kwargs)
    return session.client("s3")


def upload_prediction_to_s3(data: Dict[str, Any]) -> str:
    
    if not S3_BUCKET_NAME:
        print("S3_BUCKET_NAME no está definido. No se sube a S3.")
        return ""

    s3 = get_s3_client()

    key = f"predictions/pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.json"

    body = json.dumps(data)

    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )

    print(f"Predicción guardada en s3://{S3_BUCKET_NAME}/{key}")
    return key

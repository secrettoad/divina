from divina.divina.aws import aws_backoff
import json
import os
import sys
from botocore.exceptions import ClientError


def import_data(vision_s3_client, source_s3_client, vision_role_name):
    user_policy = {
        "Effect": "Allow",
        "Principal": {
            "AWS": "arn:aws:iam::{}:role/{}".format(
                os.environ["ACCOUNT_NUMBER"], vision_role_name
            )
        },
        "Action": ["s3:GetBucketLocation", "s3:ListBucket", "s3:GetObject"],
        "Resource": [
            "arn:aws:s3:::{}".format(os.environ["IMPORT_BUCKET"]),
            "arn:aws:s3:::{}/*".format(os.environ["IMPORT_BUCKET"]),
        ],
    }

    sys.stdout.write("Granting Divina access to imported data...\n")
    try:
        bucket_policy = json.loads(
            aws_backoff.get_bucket_policy(
                source_s3_client, bucket=os.environ["IMPORT_BUCKET"]
            )["Policy"]
        )
        if not user_policy in bucket_policy["Statement"]:
            bucket_policy["Statement"].append(user_policy)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucketPolicy":
            bucket_policy = {"Statement": [user_policy]}
        else:
            raise e

    aws_backoff.put_bucket_policy(
        source_s3_client,
        bucket=os.environ["IMPORT_BUCKET"],
        policy=json.dumps(bucket_policy),
    )

    sys.stdout.write("Creating Divina cloud storage...\n")
    try:
        if vision_s3_client._client_config.region_name == "us-east-1":
            aws_backoff.create_bucket(
                vision_s3_client,
                bucket="coysu-divina-prototype-visions",
                createBucketConfiguration={
                    "LocationConstraint": vision_s3_client._client_config.region_name
                },
            )
        else:
            aws_backoff.create_bucket(
                vision_s3_client,
                bucket="coysu-divina-prototype-visions",
                createBucketConfiguration={
                    "LocationConstraint": vision_s3_client._client_config.region_name
                },
            )

    except Exception as e:
        raise e
    try:
        source_objects = aws_backoff.list_objects(
            source_s3_client, bucket=os.environ["IMPORT_BUCKET"]
        )
    except KeyError as e:
        raise e

    sys.stdout.write("Importing data...\n")
    for file in source_objects:
        aws_backoff.copy_object(
            copy_source={"Bucket": os.environ["IMPORT_BUCKET"], "Key": file["Key"]},
            bucket="coysu-divina-prototype-visions",
            key="{}/data/{}".format(os.environ["VISION_ID"], file["Key"]),
            s3_client=vision_s3_client,
        )

    return None

import boto3
import json
import os
import datetime


IAM_PATH = 'AWS_IAM'
time_hash = '_19099290972'

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), IAM_PATH, 'S3_SOURCE_IAM')) as f:
    policy2_json = json.dumps(json.load(f))
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), IAM_PATH, 'S3_ASSUME_SOURCE_ROLE')) as f:
    policy_assume_role_json = json.dumps(json.load(f))

source_session = boto3.Session(
    aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_KEY'],
)

source_iam = source_session.client('iam')

role = source_iam.create_role(
    Path='/',
    RoleName='s3SourceRole' + time_hash,
    AssumeRolePolicyDocument=policy_assume_role_json,
    Description='role for accessing data in source s3'
)

# creates role in source account that has s3 permissions
policy_step2 = source_iam.create_policy(
    PolicyName='s3SourceRole' + time_hash,
    PolicyDocument=policy2_json
)
source_iam.attach_role_policy(
    RoleName='s3SourceRole' + time_hash, PolicyArn=policy_step2['Policy']['Arn'])

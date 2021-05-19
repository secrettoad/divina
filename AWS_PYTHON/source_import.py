import boto3
import json
import os
import datetime
import io
from botocore.exceptions import ClientError


####TODO debug assumerole error. happens when played at full speed.WORKS WHEN YOU DEBUG STEP BY STEP
####TODO abtract rootish from role jsons - would need to configure a rootish or the import role in source accounts on creation
####TODO abstract bucket names


IAM_PATH = 'AWS_IAM'
time_hash = '_19099290972'


import_session = boto3.Session(aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                               aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
import_iam = import_session.client('iam')


with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), IAM_PATH, 'S3_IMPORT_IAM')) as f:
    policy1_json = json.dumps(json.load(f))
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), IAM_PATH, 'S3_ASSUME_IMPORT_ROLE')) as f:
    policy_assume_role_json = json.dumps(json.load(f))

# creates role in source account to access S3 and to be assumed in vision account.
try:
    role = import_iam.create_role(
        Path='/',
        RoleName='s3ImportRole' + time_hash,
        AssumeRolePolicyDocument=policy_assume_role_json,
        Description='role for populating step2 s3 bucket from step1'
    )
except ClientError as e:
    if e.response['Error']['Code'] == 'EntityAlreadyExists':
        print(e)

# creates role in vision account that has s3 permissions
policy_step1 = import_iam.create_policy(
    PolicyName='s3ImportRole' + time_hash,
    PolicyDocument=policy1_json
)

import_iam.attach_role_policy(
    RoleName='s3ImportRole' + time_hash, PolicyArn=policy_step1['Policy']['Arn'])

sts_client = boto3.client('sts', aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                          aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
# Call the assume_role method of the STSConnection object and pass the role
# ARN and a role session name.
assumed_source_role = sts_client.assume_role(
    RoleArn="arn:aws:iam::169491045780:role/s3SourceRole" + time_hash,
    RoleSessionName="AssumeRoleSession1"
)

# From the response that contains the assumed role, get the temporary
# credentials that can be used to make subsequent API calls
source_credentials = assumed_source_role['Credentials']

# Use the temporary credentials that AssumeRole returns to make a
# connection to Amazon S3
source_s3 = boto3.client(
    's3',
    aws_access_key_id=source_credentials['AccessKeyId'],
    aws_secret_access_key=source_credentials['SecretAccessKey'],
    aws_session_token=source_credentials['SessionToken'],
)

assumed_import_role = sts_client.assume_role(
    RoleArn="arn:aws:iam::169491045780:role/s3ImportRole" + time_hash,
    RoleSessionName="AssumeRoleSession2"
)

# From the response that contains the assumed role, get the temporary
# credentials that can be used to make subsequent API calls
import_credentials = assumed_import_role['Credentials']

# Use the temporary credentials that AssumeRole returns to make a
# connection to Amazon S3
import_s3 = boto3.client(
    's3',
    aws_access_key_id=import_credentials['AccessKeyId'],
    aws_secret_access_key=import_credentials['SecretAccessKey'],
    aws_session_token=import_credentials['SessionToken'],
)

source_objects = source_s3.list_objects(Bucket='lambdaschool-coysu')['Contents']
try:
    import_s3.create_bucket(Bucket='lambdaschool-coysu2',
                                 CreateBucketConfiguration={
                                     'LocationConstraint': 'us-west-2'
                                 })
except ClientError as e:
    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
        print(e)

for file in source_objects:
    object_response = source_s3.get_object(
        Bucket='lambdaschool-coysu',
        Key=file['Key']
    )
    import_s3.put_object(
        ACL='public-read',
        Body=io.BytesIO(object_response['Body'].read()),
        Bucket='lambdaschool-coysu2',
        Key=file['Key']
    )

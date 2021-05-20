import boto3
import json
import os
import datetime
import io
from botocore.exceptions import ClientError
from AWS_PYTHON import source_import
import backoff


####TODO abtract rootish from role jsons - use os.path.expandvars
####TODO abstract bucket names


def vision_setup():
    iam_path = 'AWS_IAM'
    time_hash = str(datetime.datetime.now().timestamp())

    source_import.source_import(iam_path=iam_path, time_hash=time_hash)

    import_session = boto3.Session(aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                                   aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
    import_iam = import_session.client('iam')

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_IMPORT_IAM')) as f:
        policy1_json = os.path.expandvars(json.dumps(json.load(f)))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_ASSUME_IMPORT_ROLE')) as f:
        policy_assume_role_json = os.path.expandvars(json.dumps(json.load(f)))

    # creates role in vision account that has s3 permissions
    try:
        role = import_iam.create_role(
            Path='/',
            RoleName='s3ImportRole' + time_hash,
            AssumeRolePolicyDocument=policy_assume_role_json,
            Description='role for populating step2 s3 bucket from step1'
        )
        pass
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(e)
        else:
            raise e

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

    assumed_source_role = assume_role(sts_client=sts_client,
                                      roleArn="arn:aws:iam::{}:role/s3SourceRole".format(
                                          os.environ['ACCOUNT_NUMBER']) + time_hash,
                                      sessionName="AssumeRoleSession1"
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

    assumed_import_role = assume_role(sts_client=sts_client,
                                      roleArn="arn:aws:iam::{}:role/s3ImportRole".format(
                                          os.environ['ACCOUNT_NUMBER']) + time_hash,
                                      sessionName="AssumeRoleSession2")

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

    try:
        import_s3.create_bucket(Bucket='coysu-divina-prototype2',
                                CreateBucketConfiguration={
                                    'LocationConstraint': 'us-west-2'
                                })
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(e)
    try:
        source_objects = source_s3.list_objects(Bucket='coysu-divina-prototype')['Contents']
    except KeyError as e:
        raise e

    for file in source_objects:
        object_response = source_s3.get_object(
            Bucket='coysu-divina-prototype',
            Key=file['Key']
        )
        import_s3.put_object(
            ACL='public-read',
            Body=io.BytesIO(object_response['Body'].read()),
            Bucket='coysu-divina-prototype2',
            Key=file['Key']
        )


@backoff.on_exception(backoff.expo,
                      ClientError)
def assume_role(sts_client, roleArn, sessionName):
    assumed_import_role = sts_client.assume_role(
        RoleArn=roleArn,
        RoleSessionName=sessionName
    )
    return assumed_import_role


vision_setup()

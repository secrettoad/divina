import boto3
import json
import os
import datetime
import io
from botocore.exceptions import ClientError
import backoff


####TODO abtract rootish from role jsons - use os.path.expandvars
####TODO abstract bucket names
####Make central divina user that assumes each of the individual task users


def create_source_s3_role(boto3_session, iam_path='AWS_IAM'):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_SOURCE_IAM')) as f:
        policy2_json = os.path.expandvars(json.dumps(json.load(f)))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_SOURCE_ROLE_TRUST_POLICY')) as f:
        s3_source_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

    source_iam = boto3_session.client('iam')

    try:
        role = source_iam.create_role(
            Path='/',
            RoleName='s3SourceRole' + os.environ['VISION_ID'],
            AssumeRolePolicyDocument=s3_source_role_trust_policy,
            Description='role for accessing data in source s3'
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(e)

    # creates role in source account that has s3 permissions
    policy_step2 = source_iam.create_policy(
        PolicyName='s3SourceRole' + os.environ['VISION_ID'],
        PolicyDocument=policy2_json
    )
    source_iam.attach_role_policy(
        RoleName='s3SourceRole' + os.environ['VISION_ID'], PolicyArn=policy_step2['Policy']['Arn'])

    return role


def vision_setup(source_session, source_bucket, vision_session, region):
    source_s3 = source_session.client('s3')
    vision_s3 = vision_session.client('s3')

    try:
        vision_s3.create_bucket(Bucket='coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                                CreateBucketConfiguration={
                                    'LocationConstraint': region
                                })
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(e)
    try:
        source_objects = source_s3.list_objects(Bucket=source_bucket)['Contents']
    except KeyError as e:
        raise e

    for file in source_objects:
        object_response = source_s3.get_object(
            Bucket=source_bucket,
            Key=file['Key']
        )
        vision_s3.put_object(
            ACL='public-read',
            Body=io.BytesIO(object_response['Body'].read()),
            Bucket='coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
            Key=file['Key']
        )


def create_vision_iam_role(boto3_session, iam_path='AWS_IAM'):
    import_iam = boto3_session.client('iam')

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'DIVINA_IAM')) as f:
        policy1_json = os.path.expandvars(json.dumps(json.load(f)))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'DIVINA_ROLE_TRUST_POLICY')) as f:
        s3_import_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

    try:
        role = import_iam.create_role(
            Path='/',
            RoleName='visionRole' + os.environ['VISION_ID'],
            AssumeRolePolicyDocument=s3_import_role_trust_policy,
            Description='role for populating step2 s3 bucket from step1'
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(e)
        else:
            raise e

    policy_step1 = import_iam.create_policy(
        PolicyName='visionRole' + os.environ['VISION_ID'],
        PolicyDocument=policy1_json
    )

    import_iam.attach_role_policy(
        RoleName='visionRole' + os.environ['VISION_ID'], PolicyArn=policy_step1['Policy']['Arn'])

    return role


def setup_emr_cluster(vision_id, region='us-east-1'):
    emr_client = boto3.client('emr', region_name=region)

    response = emr_client.run_job_flow(
        Name="divina_cluster",
        ReleaseLabel='emr-5.12.0',
        Instances={
            'MasterInstanceType': 'm4.xsmall',
            'SlaveInstanceType': 'm4.xsmall',
            'InstanceCount': 3,
            'KeepJobFlowAliveWhenNoSteps': False,
            'TerminationProtected': False
        },
        VisibleToAllUsers=True,
        JobFlowRole='EMR_EC2_DefaultRole',
        ServiceRole='EMR_DefaultRole'
    )


@backoff.on_exception(backoff.expo,
                      ClientError)
def assume_role(sts_client, role_arn, session_name):
    assumed_import_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name
    )
    return assumed_import_role


def create_vision(create_source_role=False, create_vision_role=False, region='us-east-2'):
    os.environ['VISION_ID'] = str(datetime.datetime.now().timestamp())


    source_session = boto3.session.Session(aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'],
                                           region_name=region)

    vision_session = boto3.session.Session(aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name=region)

    if create_source_role:
        # creates role in source account that has s3 permissions

        source_s3_role = create_source_s3_role(boto3_session=source_session)

        source_sts_client = source_session.client('sts', aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                                  aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'])
        # Call the assume_role method of the STSConnection object and pass the role
        # ARN and a role session name.

        assumed_source_role = assume_role(sts_client=source_sts_client,
                                          role_arn="arn:aws:iam::{}:role/{}".format(
                                              os.environ['SOURCE_ACCOUNT_NUMBER'], source_s3_role['Role']['RoleName']),
                                          session_name="AssumeRoleSession1"
                                          )

        # From the response that contains the assumed role, get the temporary
        # credentials that can be used to make subsequent API calls
        source_credentials = assumed_source_role['Credentials']

        # Use the temporary credentials that AssumeRole returns to make a
        # connection to Amazon S3
        source_session = boto3.session.Session(
            aws_access_key_id=source_credentials['AccessKeyId'],
            aws_secret_access_key=source_credentials['SecretAccessKey'],
            aws_session_token=source_credentials['SessionToken'], region_name=region
        )

    if create_vision_role:
        vision_role = create_vision_iam_role(boto3_session=vision_session, )

        vision_sts_client = vision_session.client('sts', aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                                  aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'])

        assumed_vision_role = assume_role(sts_client=vision_sts_client,
                                          role_arn="arn:aws:iam::{}:role/{}".format(
                                              os.environ['ACCOUNT_NUMBER'], vision_role['Role']['RoleName']),
                                          session_name="AssumeRoleSession2")

        # From the response that contains the assumed role, get the temporary
        # credentials that can be used to make subsequent API calls
        vision_credentials = assumed_vision_role['Credentials']

        # Use the temporary credentials that AssumeRole returns to make a
        # connection to Amazon S3
        vision_session = boto3.session.Session(
            aws_access_key_id=vision_credentials['AccessKeyId'],
            aws_secret_access_key=vision_credentials['SecretAccessKey'],
            aws_session_token=vision_credentials['SessionToken'], region_name=region
        )

    vision_setup(source_session=source_session, vision_session=vision_session, source_bucket='coysu-divina-prototype',
                 region=region)


create_vision(create_source_role=True, create_vision_role=True)

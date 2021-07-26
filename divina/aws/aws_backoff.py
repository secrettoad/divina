from botocore.exceptions import ClientError
import backoff
import paramiko
import sys


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def create_security_group(description, group_name, vpc_id, ec2_client):
    return ec2_client.create_security_group(
        Description=description,
        GroupName=group_name,
        VpcId=vpc_id
    )


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def authorize_security_group_ingress(group_id, ip_permissions, ec2_client):
    ec2_client.authorize_security_group_ingress(
        GroupId=group_id,
        IpPermissions=ip_permissions

    )


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def describe_security_groups(filters, ec2_client):
    return ec2_client.describe_security_groups(
        Filters=filters
    )


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def list_steps(emr_client, cluster_id):
    return emr_client.list_steps(ClusterId=cluster_id)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_ec2_waiter(step, ec2_client):
    return ec2_client.get_waiter(step)


@backoff.on_exception(backoff.expo,
                      ClientError, max_tries=10)
def copy_object(copy_source, bucket, key, s3_client):
    return s3_client.copy_object(CopySource=copy_source, Bucket=bucket,
                                 Key=key)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def describe_instances(instance_ids, ec2_client):
    return ec2_client.describe_instances(InstanceIds=instance_ids)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def import_key_pair(
        key_name,
        public_key_material, ec2_client
):
    return ec2_client.import_key_pair(KeyName=key_name, PublicKeyMaterial=public_key_material)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def stop_instances(instance_ids, ec2_client):
    ec2_client.stop_instances(InstanceIds=instance_ids)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_emr_waiter(emr_client, step):
    waiter = emr_client.get_waiter(step)
    return waiter


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def delete_emr_cluster(emr_cluster, emr_client):
    emr_client.terminate_job_flows(JobFlowIds=[emr_cluster['JobFlowId']])


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_products(pricing_client, products_params):
    return pricing_client.get_products(**products_params)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_s3_object_size(s3_client, key, bucket):
    response = s3_client.head_object(Bucket=bucket, Key=key)
    size = response['ContentLength']
    return size


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def assume_role(sts_client, role_arn, session_name):
    assumed_import_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name
    )
    return assumed_import_role


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def create_bucket(s3_client, bucket, createBucketConfiguration):
    try:
        s3_client.create_bucket(Bucket=bucket,
                                CreateBucketConfiguration=createBucketConfiguration)
    except ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
            raise e
        else:
            sys.stdout.write('Cloud storage already exists...\n')


@backoff.on_exception(backoff.expo,
                      ClientError, max_tries=10)
def list_objects(s3_client, bucket, prefix=None):
    if prefix:
        objects = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    else:
        objects = s3_client.list_objects(Bucket=bucket)
    if not 'Contents' in objects:
        raise Exception('No data to import. Upload data to bucket: {} and then retry.'.format(bucket))
    return objects['Contents']


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def upload_file(s3_client, bucket, key, body):
    return s3_client.put_object(
        Bucket=bucket,
        Key=key, Body=body)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_bucket_policy(s3_client, bucket):
    return s3_client.get_bucket_policy(Bucket=bucket)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def put_bucket_policy(s3_client, bucket, policy):
    return s3_client.put_bucket_policy(Bucket=bucket, Policy=policy)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def create_role(iam_client, policy_document, trust_policy_document, role_name, policy_name, description):
    try:
        role = iam_client.get_role(RoleName=role_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            sys.stdout.write('Divina role: {} not found. Creating now.\n'.format(role_name))
            role = iam_client.create_role(
                Path='/',
                RoleName=role_name,
                AssumeRolePolicyDocument=trust_policy_document,
                Description=description
            )
        else:
            raise e

    try:
        iam_client.get_policy(
            PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name)
        )
        policy_version = iam_client.create_policy_version(
            PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name),
            PolicyDocument=policy_document,
            SetAsDefault=True
        )

        iam_client.attach_role_policy(
            RoleName=role_name, PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name))

        iam_client.delete_policy_version(
            PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name),
            VersionId='v{}'.format(int(policy_version['PolicyVersion']['VersionId'][1:]) - 1)
        )

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            sys.stdout.write('Divina policy: {} not found. Creating now.\n'.format(policy_name))
            iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=policy_document
            )
        else:
            raise e

    iam_client.attach_role_policy(
        RoleName=role_name, PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name))

    return role
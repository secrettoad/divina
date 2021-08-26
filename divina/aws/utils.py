from .aws_backoff import *
import json
from pkg_resources import resource_filename
import os
import subprocess


def create_emr_roles(boto3_session):
    iam_client = boto3_session.client("iam")

    if not all(
        x in [r["RoleName"] for r in iam_client.list_roles()["Roles"]]
        for x in ["EMR_EC2_DefaultRole", "EMR_DefaultRole"]
    ):
        subprocess.run(["aws", "emr", "create-default-roles"])

    return iam_client.list_roles()


def create_modelling_emr(
    emr_client,
    worker_profile="EMR_EC2_DefaultRole",
    driver_role="EMR_DefaultRole",
    keep_instances_alive=False,
    ec2_key=None,
):
    if keep_instances_alive:
        on_failure = "CANCEL_AND_WAIT"
    else:
        on_failure = "TERMINATE_CLUSTER"
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config/emr_config_spark.json"
        ),
        "r",
    ) as f:
        if ec2_key:
            emr_config = json.loads(
                os.path.expandvars(
                    json.dumps(json.load(f))
                    .replace("${WORKER_PROFILE}", worker_profile)
                    .replace("${DRIVER_ROLE}", driver_role)
                    .replace("${EMR_ON_FAILURE}", on_failure)
                ).replace("${EC2_KEYNAME}", ec2_key)
            )
        else:
            emr_config = json.loads(
                os.path.expandvars(
                    json.dumps(json.load(f))
                    .replace("${WORKER_PROFILE}", worker_profile)
                    .replace("${DRIVER_ROLE}", driver_role)
                    .replace("${EMR_ON_FAILURE}", on_failure)
                )
            )

    cluster = emr_client.run_job_flow(**emr_config["emr_config"])
    steps = list_steps(emr_client, cluster["JobFlowId"])
    last_step_id = steps["Steps"][0]["Id"]
    emr_waiter = get_emr_waiter(emr_client, "step_complete")
    emr_waiter.wait(
        ClusterId=cluster["JobFlowId"],
        StepId=last_step_id,
        WaiterConfig={"Delay": 30, "MaxAttempts": 120},
    )
    return cluster


def run_command_emr(emr_client, cluster_id, keep_instances_alive, args):
    if keep_instances_alive:
        on_failure = "CANCEL_AND_WAIT"
    else:
        on_failure = "TERMINATE_CLUSTER"
    steps = emr_client.add_job_flow_steps(
        JobFlowId=cluster_id,
        ###TODO start here and translate this into using the cli somehow
        Steps=[
            {
                "Name": "vision_script",
                "ActionOnFailure": on_failure,
                "HadoopJarStep": {"Jar": "command-runner.jar", "Args": args},
            }
        ],
    )
    emr_waiter = get_emr_waiter(emr_client, "step_complete")
    emr_waiter.wait(
        ClusterId=cluster_id,
        StepId=steps["StepIds"][0],
        WaiterConfig={"Delay": 60, "MaxAttempts": 120},
    )
    return steps


def ec2_pricing(pricing_client, region_name, filter_params=None):
    products_params = {
        "ServiceCode": "AmazonEC2",
        "Filters": [
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {
                "Type": "TERM_MATCH",
                "Field": "location",
                "Value": "{}".format(get_region_name(region_name)),
            },
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "shared"},
        ],
    }
    if filter_params:
        products_params["Filters"] = products_params["Filters"] + filter_params
    while True:
        response = get_products(pricing_client, products_params)
        yield from [i for i in response["PriceList"]]
        if "NextToken" not in response:
            break
        products_params["NextToken"] = response["NextToken"]
    return response


def get_region_name(region_code):
    default_region = "EU (Ireland)"
    endpoint_file = resource_filename("botocore", "data/endpoints.json")
    try:
        with open(endpoint_file, "r") as f:
            data = json.load(f)
        return data["partitions"][0]["regions"][region_code]["description"]
    except IOError:
        return default_region


def unnest_ec2_price(product):
    od = product["terms"]["OnDemand"]
    id1 = list(od)[0]
    id2 = list(od[id1]["priceDimensions"])[0]
    return {
        od[id1]["priceDimensions"][id2]["unit"]
        + "_USD": od[id1]["priceDimensions"][id2]["pricePerUnit"]["USD"]
    }


def create_divina_role(divina_session):
    divina_iam = divina_session.client("iam")
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "divina_iam_policy.json",
        )
    ) as f:
        divina_policy = os.path.expandvars(json.dumps(json.load(f)))
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "divina_trust_policy.json",
        )
    ) as f:
        divina_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

    try:
        divina_iam.remove_role_from_instance_profile(
            InstanceProfileName="divina-instance-profile", RoleName="divina-role"
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            pass
        else:
            raise e
    try:
        divina_iam.detach_role_policy(
            RoleName="divina-role",
            PolicyArn="arn:aws:iam::{}:policy/divina-role-policy".format(
                os.environ["ACCOUNT_NUMBER"]
            ),
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            pass
        else:
            raise e
    try:
        divina_iam.delete_role(RoleName="divina-role")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            pass
        else:
            raise e
    divina_role = create_role(
        divina_iam,
        divina_policy,
        divina_role_trust_policy,
        "divina-role",
        "divina-role-policy",
        "role for coysu divina",
    )

    try:
        divina_iam.create_instance_profile(
            InstanceProfileName="divina-instance-profile"
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            pass
        else:
            raise e
    divina_iam.add_role_to_instance_profile(
        InstanceProfileName="divina-instance-profile",
        RoleName="divina-role",
    )

    return divina_role


def grant_bucket_access(access_role_name, bucket_name, boto3_session):
    s3_client = boto3_session.client("s3")
    user_policy = {
        "Effect": "Allow",
        "Principal": {
            "AWS": "arn:aws:iam::{}:role/{}".format(
                os.environ["ACCOUNT_NUMBER"], access_role_name
            )
        },
        "Action": ["s3:GetBucketLocation", "s3:ListBucket", "s3:GetObject"],
        "Resource": [
            "arn:aws:s3:::{}".format(bucket_name),
            "arn:aws:s3:::{}/*".format(bucket_name),
        ],
    }
    try:
        bucket_policy = json.loads(
            get_bucket_policy(s3_client, bucket=bucket_name)["Policy"]
        )
        if not user_policy in bucket_policy["Statement"]:
            bucket_policy["Statement"].append(user_policy)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucketPolicy":
            bucket_policy = {"Statement": [user_policy]}
        else:
            raise e

    put_bucket_policy(
        s3_client,
        bucket=bucket_name,
        policy=json.dumps(bucket_policy),
    )

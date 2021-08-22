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


def create_vision_role(vision_session):
    vision_iam = vision_session.client("iam")
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
        vision_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

    vision_role = create_role(
        vision_iam,
        divina_policy,
        vision_role_trust_policy,
        "divina-vision-role",
        "divina-vision-role-policy",
        "role for coysu divina",
    )

    vision_instance_profile = vision_iam.create_instance_profile(
        InstanceProfileName="divina-vision-instance-profile"
    )

    vision_iam.add_role_to_instance_profile(
        InstanceProfileName="divina-vision-instance-profile",
        RoleName="divina-vision-role",
    )

    return vision_role, vision_instance_profile

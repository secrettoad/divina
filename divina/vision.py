import boto3
import os
import datetime
import sys
import json
from divina.divina.aws import aws_backoff
from .errors import InvalidDataDefinitionException
import pathlib
from .aws.utils import (
    create_emr_roles,
    create_modelling_emr,
    run_command_emr,
    create_divina_role,
)


####TODO abtract rootish from role jsons - use os.path.expandvars


def validate_vision_definition(vision_definition):
    if not "time_index" in vision_definition:
        raise InvalidDataDefinitionException(
            "required field time_index not found in data definition"
        )
    if not "target" in vision_definition:
        raise InvalidDataDefinitionException(
            "required field target not found in data definition"
        )
    if "time_validation_splits" in vision_definition:
        if not type(vision_definition["time_validation_splits"]) == list:
            raise InvalidDataDefinitionException(
                "time_validation_splits must be a list of date-like strings"
            )
        elif not all(
            [type(x) == str for x in vision_definition["time_validation_splits"]]
        ):
            raise InvalidDataDefinitionException(
                "time_validation_splits must be a list of date-like strings"
            )
    if "time_horizons" in vision_definition:
        if not type(vision_definition["time_horizons"]) == list:
            raise InvalidDataDefinitionException(
                "time_horizons must be a list of integers"
            )
        elif not all([type(x) == int for x in vision_definition["time_horizons"]]):
            raise InvalidDataDefinitionException(
                "time_horizons must be a list of integers"
            )


def create_vision(
    s3_fs,
    divina_directory,
    worker_profile="EMR_EC2_DefaultRole",
    driver_role="EMR_DefaultRole",
    region="us-east-2",
    ec2_keypair_name=None,
    vision_definition=None,
    keep_instances_alive=False,
    verbosity=0,
    commit="main",
):
    os.environ["VISION_ID"] = str(round(datetime.datetime.now().timestamp()))
    os.environ["DIVINA_BUCKET"] = "coysu-divina-prototype-visions"

    sys.stdout.write("Authenticating to the cloud...\n")
    vision_session = boto3.session.Session(region_name=region)

    if not worker_profile and driver_role:
        sys.stdout.write("Creating spark driver and executor cloud roles...\n")
        create_emr_roles(vision_session)

    with s3_fs.open(
        pathlib.Path(
            divina_directory,
            "{}/vision_definition.json".format(os.environ["VISION_ID"]),
        ),
        "w+",
    ) as f:
        json.dump(vision_definition, f)

    create_divina_role(divina_session=vision_session)

    sys.stdout.write("Creating forecasts...\n")
    emr_client = vision_session.client("emr")
    if ec2_keypair_name:
        emr_cluster = create_modelling_emr(
            emr_client=emr_client,
            worker_profile=worker_profile,
            driver_role=driver_role,
            keep_instances_alive=keep_instances_alive,
            ec2_key=ec2_keypair_name,
        )
    else:
        emr_cluster = create_modelling_emr(
            emr_client=emr_client,
            worker_profile=worker_profile,
            driver_role=driver_role,
            keep_instances_alive=keep_instances_alive,
        )

    run_command_emr(
        emr_client=emr_client,
        cluster_id=emr_cluster["JobFlowId"],
        keep_instances_alive=keep_instances_alive,
        args=[
            "divina",
            "train",
            "--data_definition",
            "/home/hadoop/data_definition.json",
            "",
        ],
    )
    steps = aws_backoff.list_steps(emr_client, emr_cluster["JobFlowId"])
    last_step_id = steps["Steps"][0]["Id"]
    emr_waiter = aws_backoff.get_emr_waiter(emr_client, "step_complete")
    emr_waiter.wait(
        ClusterId=emr_cluster["JobFlowId"],
        StepId=last_step_id,
        WaiterConfig={"Delay": 30, "MaxAttempts": 120},
    )

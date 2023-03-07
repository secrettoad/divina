#!/usr/bin/env bash

sleep 60

if curl -s -m 5 http://169.254.169.254/latest/dynamic/instance-identity/document | grep -q availabilityZone
then
  echo 'EC2 detected'
  IP=$(curl http://169.254.169.254/latest/meta-data/public-ipv4);
  export PREFECT_ORION_API_HOST=$IP
fi
prefect orion start --host=0.0.0.0


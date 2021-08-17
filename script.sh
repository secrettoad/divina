#!/bin/bash
               sudo yum install python3 -y;
               sudo yum install git -y;
               sudo python3 -m pip install git+https://git@github.com/secrettoad/divina.git@main;
               sudo chown -R ec2-user /home/ec2-user;
               divina dataset build s3://divina-test/data s3://divina-test/dataset test_e2e_1

#!/usr/bin/env bash

# create /var/log/auth.log if not exist
if [[ ! -f /var/log/auth.log ]]
then
    touch /var/log/auth.log
fi

# sleep forever
#sleep infinity

# link auth.log to container log
tail -f /var/log/auth.log
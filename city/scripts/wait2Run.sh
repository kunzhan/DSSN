#!/bin/bash

# Set the PID of the process you want to monitor
pid=1011882

# Set the command you want to run when the process ends
script_command="bash ./scripts/trainCity0.sh 8 8285"

# Monitor the process using ps
while ps -p $pid > /dev/null; do
  sleep 1
done

# The process has ended, so run the script
$script_command

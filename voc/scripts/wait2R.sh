#!/bin/bash

# Set the PID of the process you want to monitor
pid=554287

# Set the command you want to run when the process ends
script_command="bash ./scripts/voc.sh 4 8385"

# Monitor the process using ps
while ps -p $pid > /dev/null; do
  sleep 1
done

# The process has ended, so run the script
$script_command

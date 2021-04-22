#!/usr/bin/env bash
source ./config.properties
echo ${source}

sudo docker stop ${container_name}

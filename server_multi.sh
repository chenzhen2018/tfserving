#!/usr/bin/env bash
source ./config.properties

# shellcheck disable=SC2154
sudo docker run \
-p 8500:8500 \
--mount type=bind,source=${source},target=${target} \
--name=${container_name} \
-t ${images_name} \
--model_config=${model_config}

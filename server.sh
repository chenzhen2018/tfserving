#!/usr/bin/env bash
source ./config.properties

# shellcheck disable=SC2154
sudo docker run \
-p 8500:8500 \
--mount type=bind,source=${source},target=${target} \
-e MODEL_NAME=${model_name} \
--name=${container_name} \
-t ${images_name}

#!/bin/bash

if [[ $# -eq 0 ]]; then
    echo -e "No parameters found. "
    exit 1
fi
if [[ -d "$1" ]]; then
    dir=$( readlink -f "$1" )
    echo "Executing in $dir. "
    docker run -it --rm --gpus all \
      -p 8888:8888 \
      --volume "$dir:$dir" \
      --workdir "$dir" --env HOME="$dir" \
      --name mt mt:latest jupyter lab --ServerApp.token='' --ServerApp.password='' --allow-root --ip 0.0.0.0
else
    echo -e "Argument should be a directory. "
fi


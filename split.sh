#!/bin/bash
set -e

FOLDER="${1:-pwd}"
cd $FOLDER

INPUT="${2:-$(ls ./*.m4b| head -1)}"
echo $INPUT

docker run -it --rm -u $(id -u):$(id -g) -v "$FOLDER":/mnt sandreas/m4b-tool:latest split ./$INPUT

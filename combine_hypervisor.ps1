$FOLDER=$args[0]
$NAME=Split-Path "$FOLDER" -Leaf
$SPLIT_DIR=$FOLDER +"\"+ $NAME + "_splitted\"
Write-Host "$FOLDER"
Write-Host $NAME
Write-Host $SPLIT_DIR

# Get-ChildItem -Path "$FOLDER" -Directory |
# Sort-Object LastWriteTime |
# Select-Object -ExpandProperty Name -Last 1

# function m4b-tool {
#   docker run -it --rm -v "$FOLDER":/mnt sandreas/m4b-tool:latest $args
# }
docker run -it --rm -v $FOLDER:/mnt sandreas/m4b-tool:latest merge $SPLIT_DIR --output-file=./$NAME.m4b -vvv
# m4b-tool "./${$NAME}_splitted/" --output-file="./$NAME.m4b" -vvv

#working from windows
# docker run -it --rm -v ${PWD}:/mnt sandreas/m4b-tool:latest merge "./a/" --output-file="./かがみの孤城.m4b" --jobs=4
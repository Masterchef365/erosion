#!/bin/sh

for x in *.comp; do
    glslc -g -O $x -o $x.spv &
done
wait

#!/bin/sh

for x in *.frag *.vert; do
    glslc -g -O $x -o $x.spv &
done
wait

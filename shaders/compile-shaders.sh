#!/bin/sh

compile() {
    glslc -g -O $1 -o $1.spv
}

compile unlit.vert
compile heightmap.vert
compile unlit.frag
compile unlit_tex.frag

#!/usr/bin/env bash
make clean build

make run ARGS="-input=data/lena.pgm"

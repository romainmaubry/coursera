#!/usr/bin/env bash
 make clean build
 make  run
 make  run ARGS="--input=data/casablanca.pgm"
 make  run ARGS="--input=data/mona_lisa.pgm"
 make  run ARGS="--input=data/baboon.pgm"

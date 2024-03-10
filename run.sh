#!/usr/bin/env bash
 make clean build
 make  run
 make  run ARGS="--input=data/casablanca.pgm"
 make  run ARGS="--input=data/monalisa.pgm"
 make  run ARGS="--input=data/baboon.pgm"

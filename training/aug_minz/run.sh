#! /bin/sh

nohup python fit.py fit_gen trc valc --epochs 100 --batch_size 64 --gpu_id 5 > Log/nohup.out &

#! /bin/sh

for i in $(seq 0 9)
    do
        nohup python _1_downsample.py run "$i" > "Log/1_nohup.$i.out" &
    done


#! /bin/sh

for i in $(seq 0 9)
    do
        nohup python _2_transpose.py run "$i" > "Log/2_nohup.$i.out" &
    done


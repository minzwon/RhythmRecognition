#! /bin/sh

for i in $(seq 0 9)
    do
        nohup python _6_fool_transpose.py run "$i" > "Log/6_nohup.$i.out_19p" &
    done


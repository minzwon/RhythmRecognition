#! /bin/sh

for i in $(seq 0 9)
    do
        nohup python _5_fool.py run "$i" > "Log/5_nohup.$i.out" &
    done


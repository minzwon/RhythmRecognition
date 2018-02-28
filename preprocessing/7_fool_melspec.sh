#! /bin/sh

for i in $(seq 0 9)
    do
        nohup python _7_fool_melspec.py run "$i" > "Log/7_trp_nohup.$i.out" &
        nohup python _7_fool_melspec.py run "$i" True > "Log/7_16k_nohup.$i.out" &
    done


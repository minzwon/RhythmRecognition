#! /bin/sh

for i in $(seq 0 9)
    do
        nohup python _3_get_melspec.py run "$i" > "Log/3_trp_nohup.$i.out" &
        nohup python _3_get_melspec.py run "$i" True > "Log/3_16k_nohup.$i.out" &
    done


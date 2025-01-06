#!/bin/sh

# Settings 
time_idx="Timestamp"
categorical_idxs="Dst Port"
continuous_idxs="Flow Duration,Total Length of Fwd Packet,Total Length of Bwd Packet,Fwd Header Length,Bwd Header Length,Flow IAT Mean"
label_col="Label"
freq="10s"
width=3
k=48
FB=128
init_len=6 # width x 2
input_fname="partial_cci_18"

# CyberCScope
python3 main.py \
    --input_fpath "./_dat/"$input_fname".csv.gz" \
    --out_dir "./_out/"$input_fname \
    --time_idx $time_idx \
    --categorical_idxs "$categorical_idxs" \
    --continuous_idxs "$continuous_idxs" \
    --label_col $label_col \
    --freq $freq \
    --width $width \
    --k $k \
    --FB $FB \
    --init_len $init_len
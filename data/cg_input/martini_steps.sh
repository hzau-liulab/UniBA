#!/bin/bash


## Activate conda environment python2 and run martini2
# conda activate python2

# Define path
dssp_path="./py/mkdssp"
cp /home/data/yyShen/.conda/envs/test/bin/mkdssp /home/data/yyShen/UniBA/data/cg_input/py/
pdb_folder="../../pdb_files/complex"
output_folder="../"
pair_file="../train_pair.txt"

for pair in $(sed -n '1p' "$pair_file"); do  #1,$p
    pair=$(echo "$pair" | tr -d '\r' | xargs)
    [ -z "$pair" ] && continue
    echo "Processing pair: '$pair'"

    pdbname="${pdb_folder}/${pair}.pdb"
    chain=$(echo $pair | cut -d'_' -f2 | cut -c1)
    raw_chain=$(echo $pair | cut -d'_' -f2)
    output_dir="./${pair}"

    if [ -d "$output_dir" ] && \
       [ "$(find "$output_dir" -type f | wc -l)" -gt 0 ] && \
       [ "$(find "$output_dir" -type f -size 0 | wc -l)" -eq 0 ]; then
        echo "Info: Output folder $output_dir contains only non-empty files, skip this pair."
        continue
    fi

    rm -rf "${output_dir:?}"/* 2>/dev/null
    mkdir -p "$output_dir"
    cd $output_dir || { echo "Failed to change directory to $output_dir"; return 1; }

    python2 /home/data/yyShen/UniBA/data/cg_input/py/martini.py -f $pdbname -o cg_M2.top -x cg_M2.pdb -dssp $dssp_path -p Backbone -ff martini22 -v
    ## Convert itp files for martini2
    python /home/data/yyShen/UniBA/data/cg_input/py/itpconv.py Protein_$chain.itp > cg_${raw_chain}_M2.itp

    cd ${output_folder} || { echo "Failed to change directory to $output_dir"; return 1; }

done < "$pair_file"

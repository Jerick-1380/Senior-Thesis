#!/bin/bash

declare -A total_gpu
declare -A idle_gpu
declare -A alloc_gpu

node_block=""

# Read lines and build full node blocks
while IFS= read -r line || [[ -n $line ]]; do
    if [[ $line == NodeName=* ]]; then
        if [[ -n "$node_block" ]]; then
            # Process previous block
            state=$(echo "$node_block" | grep -oP 'State=\K[^ ]+')
            gres_line=$(echo "$node_block" | grep -oP 'Gres=gpu:[^ ]+')

            IFS=',' read -ra gpus <<< "${gres_line#Gres=gpu:}"
            for g in "${gpus[@]}"; do
                if [[ $g =~ ([^:]+):([0-9]+) ]]; then
                    gpu_type="${BASH_REMATCH[1]}"
                    gpu_count="${BASH_REMATCH[2]}"
                    total_gpu[$gpu_type]=$((total_gpu[$gpu_type] + gpu_count))

                    if [[ "$state" == *IDLE* ]]; then
                        idle_gpu[$gpu_type]=$((idle_gpu[$gpu_type] + gpu_count))
                    elif [[ "$state" == *MIXED* || "$state" == *ALLOCATED* ]]; then
                        alloc_gpu[$gpu_type]=$((alloc_gpu[$gpu_type] + gpu_count))
                    fi
                fi
            done
        fi
        node_block="$line"
    else
        node_block+=$'\n'"$line"
    fi
done < <(scontrol show node)

# Process last block
if [[ -n "$node_block" ]]; then
    state=$(echo "$node_block" | grep -oP 'State=\K[^ ]+')
    gres_line=$(echo "$node_block" | grep -oP 'Gres=gpu:[^ ]+')

    IFS=',' read -ra gpus <<< "${gres_line#Gres=gpu:}"
    for g in "${gpus[@]}"; do
        if [[ $g =~ ([^:]+):([0-9]+) ]]; then
            gpu_type="${BASH_REMATCH[1]}"
            gpu_count="${BASH_REMATCH[2]}"
            total_gpu[$gpu_type]=$((total_gpu[$gpu_type] + gpu_count))

            if [[ "$state" == *IDLE* ]]; then
                idle_gpu[$gpu_type]=$((idle_gpu[$gpu_type] + gpu_count))
            elif [[ "$state" == *MIXED* || "$state" == *ALLOCATED* ]]; then
                alloc_gpu[$gpu_type]=$((alloc_gpu[$gpu_type] + gpu_count))
            fi
        fi
    done
fi

# Output the table
printf "\n%-12s %-10s %-10s %-10s\n" "GPU Type" "Total" "Idle" "Allocated"
printf "%s\n" "---------------------------------------------"
for gpu in "${!total_gpu[@]}"; do
    printf "%-12s %-10d %-10d %-10d\n" \
        "$gpu" \
        "${total_gpu[$gpu]}" \
        "${idle_gpu[$gpu]:-0}" \
        "${alloc_gpu[$gpu]:-0}"
done
echo ""
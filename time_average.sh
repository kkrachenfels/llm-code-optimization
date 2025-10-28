#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 program [args...]"
  exit 1
fi

program="$1"
shift
args=("$@")

runs=5

total_real=0
total_user=0
total_sys=0

convert_to_seconds() {
  # Convert time format XmYs or Xs to seconds as decimal
  local time_str="$1"
  if [[ $time_str =~ ^([0-9]+)m([0-9]+\.[0-9]+)s$ ]]; then
    local minutes=${BASH_REMATCH[1]}
    local seconds=${BASH_REMATCH[2]}
    local total=$(echo "$minutes * 60 + $seconds" | bc)
  elif [[ $time_str =~ ^([0-9]+\.[0-9]+)s$ ]]; then
    local total="${BASH_REMATCH[1]}"
  else
    local total="$time_str"
  fi
  # Ensure leading zero for decimal numbers less than 1
  if [[ $total =~ ^\.[0-9]+$ ]]; then
    total="0$total"
  fi
  echo "$total"
}

for i in $(seq 1 $runs); do
  # output time results
  output=$( { time "$program" "${args[@]}" 1>/dev/null; } 2>&1 )
  
  echo "Run $i output:"
  echo "$output"
  
  raw_real_time=$(echo "$output" | grep ^real | awk '{print $2}')
  raw_user_time=$(echo "$output" | grep ^user | awk '{print $2}')
  raw_sys_time=$(echo "$output" | grep ^sys | awk '{print $2}')
  
  real_time=$(convert_to_seconds "$raw_real_time")
  user_time=$(convert_to_seconds "$raw_user_time")
  sys_time=$(convert_to_seconds "$raw_sys_time")
  
  if ! [[ $real_time =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: invalid real time value: '$real_time'"
    exit 1
  fi
  if ! [[ $user_time =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: invalid user time value: '$user_time'"
    exit 1
  fi
  if ! [[ $sys_time =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: invalid sys time value: '$sys_time'"
    exit 1
  fi
  
  total_real=$(echo "$total_real + $real_time" | bc)
  total_user=$(echo "$total_user + $user_time" | bc)
  total_sys=$(echo "$total_sys + $sys_time" | bc)
done

avg_real=$(echo "scale=3; $total_real / $runs" | bc)
avg_user=$(echo "scale=3; $total_user / $runs" | bc)
avg_sys=$(echo "scale=3; $total_sys / $runs" | bc)

echo "Average times over $runs runs:"
echo "Real: $avg_real seconds"
echo "User: $avg_user seconds"
echo "Sys:  $avg_sys seconds"
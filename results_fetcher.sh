#!/bin/bash

# Define the file containing the list of IPs
IP_LIST="pcs.txt"

# Define the source directory on remote machines (adjust path as needed)
SOURCE_DIR="/home/i17pclab"

# Define the destination directory on your computer
DEST_DIR="./results/"

# Check if IP list file exists
if [[ ! -f "$IP_LIST" ]]; then
  echo "Error: IP list file '$IP_LIST' not found!"
  exit 1
fi

# Check if destination directory exists, create it if not
if [[ ! -d "$DEST_DIR" ]]; then
  mkdir "$DEST_DIR"
fi

# Loop through each IP address
while IFS=', ' read -r name; do
  # Escape spaces in the IP address for scp
  escaped_ip=$(echo "$name" | sed 's/ /\\ /g')

  # Construct the source and destination file paths
  source_file="$SOURCE_DIR/BPO_Batching/results.csv"
  dest_file="$DEST_DIR/$name"

  # Copy files using scp, handle errors
  if sshpass -p i17 scp -o ConnectTimeout=1 -o ConnectionAttempts=1 "i17pclab@$name:$source_file" "$dest_file.csv"; then
    echo "Copied files from $name to $dest_file"
  else
    echo "Error copying files from $name"
    #exit 1
  fi

done < "$IP_LIST"

echo "Finished copying files."

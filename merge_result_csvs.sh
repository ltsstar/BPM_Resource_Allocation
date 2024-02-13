#!/bin/bash

# Define the directory containing the CSV files
CSV_DIR="results"

# Define the output file name
OUTPUT_FILE="merged.csv"

# Check if the directory exists
if [[ ! -d "$CSV_DIR" ]]; then
  echo "Error: Directory '$CSV_DIR' doesn't exist!"
  exit 1
fi

# Combine all CSV files in the directory
echo "Merging CSV files..."
cat "$CSV_DIR"/*.csv > "$OUTPUT_FILE"

echo "Merged files into '$OUTPUT_FILE'."



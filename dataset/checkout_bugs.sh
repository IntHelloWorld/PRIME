#!/bin/bash

# Set the projects path
BENCHMARK_PATH="/home/qyh/DATASET/BugsInPy/projects/"
repo_path="/disk2/qyh/TreeFL/dataset/buggy_codebase"

# Find the first bug.info file and process it
find "$BENCHMARK_PATH" -name "bug.info" -type f | while read -r bug_info_file; do

    # Extract project name (3 levels up from bug.info)
    proj=$(basename "$(dirname "$(dirname "$(dirname "$bug_info_file")")")")
    echo "Processing project: $proj"

    # Extract bug ID (1 level up from bug.info)
    bug_id=$(basename "$(dirname "$bug_info_file")")
    echo "Processing bug ID: $bug_id"

    if [ $proj == "ansible" ]; then
        echo $"Processing project: $proj"
        test_output_file="$repo_path/$proj/$bug_id/$proj/bugsinpy_fail.txt"
        # if [ -f "$test_output_file" ]; then
        #     echo "Test output file already exists for $proj/$bug_id, skipping..."
        #     continue
        # fi

        if [ $proj == "pandas" ]; then
            setupfile="$repo_path/$proj/$bug_id/$proj/setup.py"
            sed -i 's/extra_compile_args = \["-Werror"\]/extra_compile_args = []/g' "$setupfile"
        fi

        # Checkout the bug
        # bugsinpy-checkout -p "$proj" -v 0 -i "$bug_id" -w "$repo_path/$proj/$bug_id"
        
        # Change to the project directory and compile
        cd "$repo_path/$proj/$bug_id/$proj"
        # bugsinpy-compile
        bugsinpy-test
    else
        echo "Skipping project: $proj"
        continue
    fi
done
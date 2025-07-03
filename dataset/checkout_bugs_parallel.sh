#!/bin/bash

# Set the projects path
BENCHMARK_PATH="/home/qyh/DATASET/BugsInPy/projects/"
repo_path="/disk2/qyh/TreeFL/dataset/buggy_codebase"

# Number of parallel processes (adjust based on your system)
PARALLEL_JOBS=4

# Function to process a single bug.info file
process_bug() {
    local bug_info_file="$1"
    
    # Extract project name (3 levels up from bug.info)
    proj=$(basename "$(dirname "$(dirname "$(dirname "$bug_info_file")")")")
    echo "Processing project: $proj"

    # Extract bug ID (1 level up from bug.info)
    bug_id=$(basename "$(dirname "$bug_info_file")")
    echo "Processing bug ID: $bug_id"

    if [ $proj == "pandas" ]; then
        echo "Processing project: $proj"
        test_output_file="$repo_path/$proj/$bug_id/$proj/bugsinpy_fail.txt"
        if [ -f "$test_output_file" ]; then
            echo "Test output file already exists for $proj/$bug_id, skipping..."
            return
        fi

        setupfile="$repo_path/$proj/$bug_id/$proj/setup.py"
        sed -i 's/extra_compile_args = \["-Werror"\]/extra_compile_args = []/g' "$setupfile"

        # Checkout the bug
        # bugsinpy-checkout -p "$proj" -v 0 -i "$bug_id" -w "$repo_path/$proj/$bug_id"
        
        # Change to the project directory and compile
        cd "$repo_path/$proj/$bug_id/$proj"
        bugsinpy-compile
        bugsinpy-test
    elif [ $proj == "scrapy" ]; then
        # skip the first bug in scrapy
        if [ "$bug_id" == "1" ]; then
            echo "Skipping first bug in $proj"
            return
        fi

        cd "$repo_path/$proj/$bug_id/$proj"
        if [ ! -d "env" ]; then
            rm -rf env
        fi

        # copy the "/disk2/qyh/TreeFL/dataset/buggy_codebase/scrapy/1/scrapy/env"
        cp -r "/disk2/qyh/TreeFL/dataset/buggy_codebase/scrapy/1/scrapy/env" "$repo_path/$proj/$bug_id/$proj/"

        bugsinpy-test
    else
        echo "Skipping project: $proj"
        return
    fi
}

# Export the function so it can be used by xargs
export -f process_bug
export BENCHMARK_PATH repo_path

# Find all bug.info files and process them in parallel
find "$BENCHMARK_PATH" -name "bug.info" -type f | xargs -I {} -P "$PARALLEL_JOBS" bash -c 'process_bug "$@"' _ {}
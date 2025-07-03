#!/bin/bash

# Set the projects path
BENCHMARK_PATH="/home/qyh/DATASET/BugsInPy/projects/"
repo_path="/disk2/qyh/TreeFL/dataset/buggy_codebase"

# Number of parallel processes (adjust based on your system)
PARALLEL_JOBS=12

# Function to process a single bug.info file
process_bug() {
    local bug_info_file="$1"
    
    # Extract project name (3 levels up from bug.info)
    proj=$(basename "$(dirname "$(dirname "$(dirname "$bug_info_file")")")")
    echo "Processing project: $proj"

    # Extract bug ID (1 level up from bug.info)
    bug_id=$(basename "$(dirname "$bug_info_file")")
    echo "Processing bug ID: $bug_id"
    
    # Change to the project directory
    cd "$repo_path/$proj/$bug_id/$proj"
    bugsinpy-callgraph
}

# Export the function so it can be used by xargs
export -f process_bug
export BENCHMARK_PATH repo_path

# Find all bug.info files and process them in parallel
find "$BENCHMARK_PATH" -name "bug.info" -type f | xargs -I {} -P "$PARALLEL_JOBS" bash -c 'process_bug "$@"' _ {}
# Wait for all background processes to finish
wait
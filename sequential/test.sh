#!/bin/bash
# Sample usage: 
# ./test.sh test/particle-density-scaling
# ./test.sh test/particle-density-scaling out

# Exit if specified directory does not exist
if [[ ! -e $1 ]]; then
    echo "Test directory does not exist: $1"
    exit 1
fi

# Compile C++ simulation code
make
echo 

# Set the extension of output files
OUTPUT_EXTENSION=${2:-out}
echo "Using output extension: .$OUTPUT_EXTENSION"

# Create output dir if it doesn't exist
OUTPUT_DIR=$1/out
if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi

echo "Executing simulation using input files..."
for (( i=1; i <= 3; ++i )); do
    for FILE_PATH in $1/*; do 
        if [ -f "$FILE_PATH" ]; then 
            FILE=$(basename -- "$FILE_PATH")
            NAME=${FILE%%.*}
            EXTENSION=${FILE##*.}

            # Run the simulation for input files
            if [ "$EXTENSION" == "in" ]; then
                perf stat -e instructions -e cycles -e context-switches -e cache-references -e cache-misses -x, -- ./main < $FILE_PATH 2>$OUTPUT_DIR/$NAME.perf$i 1>$OUTPUT_DIR/$NAME.$OUTPUT_EXTENSION$i
            #    ./main < $FILE_PATH > $OUTPUT_DIR/$NAME.$OUTPUT_EXTENSION
                echo "Output stored in $OUTPUT_DIR/$NAME.$OUTPUT_EXTENSION$i"
            fi
        fi 
    done
done
echo

python3 process.py $OUTPUT_DIR
echo "Stored processed results in $OUTPUT_DIR/results.csv"

#!/bin/bash
# tr -d '\r' < test.sh1 > test.sh

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
for FILE_PATH in $1/*; do 
    if [ -f "$FILE_PATH" ]; then 
        FILE=$(basename -- "$FILE_PATH")
        NAME=${FILE%%.*}
        EXTENSION=${FILE##*.}

        # Run the simulation for input files
        if [ "$EXTENSION" == "in" ]; then
            perf stat -e instructions -e cycles -e context-switches -e cache-references -e cache-misses -- ./main < $FILE_PATH 2>$OUTPUT_DIR/$NAME.perf 1>$OUTPUT_DIR/$NAME.$OUTPUT_EXTENSION
        #    ./main < $FILE_PATH > $OUTPUT_DIR/$NAME.$OUTPUT_EXTENSION
            echo "Output stored in $OUTPUT_DIR/$NAME.$OUTPUT_EXTENSION"
        fi
    fi 
done
echo
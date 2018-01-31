#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all bag files in the specified directory
DATASET_DIRECTORY=$1
for data in $(find ${DATASET_DIRECTORY} -name '*.h5'); do
	echo "Processing dataset file ${data}"
	
	INPUT_DATASET_FILE="${data}"
	OUTPUT_VISUALIZATION_FILE="${data%.h5}.preview.mp4"
	
	if [ -f $OUTPUT_VISUALIZATION_FILE ]; then
		echo "Output dataset ${OUTPUT_VISUALIZATION_FILE} already found"
		echo "Skipping conversion for dataset file ${INPUT_DATASET_FILE}"
		continue
	fi
	
	# Extract video preview from the HDF5 file
	python ${DIR}/visualize_hdf5_mp4.py --input=$INPUT_DATASET_FILE --output=$OUTPUT_VISUALIZATION_FILE --fs-video=2 --speed-up=7.5
	
done

echo 'All done.'


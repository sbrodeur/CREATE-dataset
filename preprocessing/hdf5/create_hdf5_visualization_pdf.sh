#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all bag files in the specified directory
DATASET_DIRECTORY=$1
for data in $(find ${DATASET_DIRECTORY} -name '*.h5'); do
	echo "Processing dataset file ${data}"
	
	INPUT_DATASET_FILE="${data}"
	OUTPUT_VISUALIZATION_FILE="${data%.h5}.preview.pdf"
	
	if [ -f $OUTPUT_VISUALIZATION_FILE ]; then
		echo "Output dataset ${OUTPUT_VISUALIZATION_FILE} already found"
		echo "Skipping conversion for dataset file ${INPUT_DATASET_FILE}"
		continue
	fi
	
	python ${DIR}/visualize_hdf5_pdf.py --input=$INPUT_DATASET_FILE --output=$OUTPUT_VISUALIZATION_FILE --dpi=300
	if ! [ -f $OUTPUT_VISUALIZATION_FILE ]; then
		echo "Could not find temporary file ${OUTPUT_VISUALIZATION_FILE}. An error probably occured during conversion."
		exit 1
	fi
	
done

echo 'All done.'


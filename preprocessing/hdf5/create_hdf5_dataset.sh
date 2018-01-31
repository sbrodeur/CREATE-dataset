#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all bag files in the specified directory
BAG_DIRECTORY=$1
for bag in $(find ${BAG_DIRECTORY} -name '*.bag'); do 
	echo "Processing bag file ${bag}"
	
	INPUT_DATASET_FILE="${bag}"
	OUTPUT_DATASET_FILE_HDF5_RAW="${bag%.bag}.h5.raw"
	OUTPUT_DATASET_FILE_HDF5_SORT="${bag%.bag}.h5.sort"
	OUTPUT_DATASET_FILE_HDF5_OPFLOW="${bag%.bag}.h5.optflow"
	OUTPUT_DATASET_FILE_HDF5="${bag%.bag}.h5"
	
	if [ -f $OUTPUT_DATASET_FILE_HDF5 ]; then
		echo "Output dataset ${OUTPUT_DATASET_FILE_HDF5} already found"
		echo "Skipping conversion for bag file ${INPUT_DATASET_FILE}"
		continue
	fi
	
	echo "Removing any existing dataset temporary files"
	rm -f $OUTPUT_DATASET_FILE_HDF5_RAW $OUTPUT_DATASET_FILE_HDF5_SORT $OUTPUT_DATASET_FILE_HDF5_OPFLOW
	
	python ${DIR}/convert_hdf5.py --input=$INPUT_DATASET_FILE --output=$OUTPUT_DATASET_FILE_HDF5_RAW --use-capture-time --use-relative-time
	if ! [ -f $OUTPUT_DATASET_FILE_HDF5_RAW ]; then
		echo "Could not find temporary file ${OUTPUT_DATASET_FILE_HDF5_RAW}. An error probably occured during conversion."
		exit 1
	fi
	
	python ${DIR}/sort_hdf5.py --input=$OUTPUT_DATASET_FILE_HDF5_RAW --output=$OUTPUT_DATASET_FILE_HDF5_SORT
	if ! [ -f $OUTPUT_DATASET_FILE_HDF5_SORT ]; then
		echo "Could not find temporary file ${OUTPUT_DATASET_FILE_HDF5_SORT}. An error probably occured during conversion."
		exit 1
	fi
	
	python ${DIR}/opflow_hdf5.py --input=$OUTPUT_DATASET_FILE_HDF5_SORT --output=$OUTPUT_DATASET_FILE_HDF5_OPFLOW --scale=0.25 --border=0.1 --grid-shape-y=12 --grid-shape-x=16 --filtering --filter-size=5 --step-size=5
	if ! [ -f $OUTPUT_DATASET_FILE_HDF5_OPFLOW ]; then
		echo "Could not find temporary file ${OUTPUT_DATASET_FILE_HDF5_OPFLOW}. An error probably occured during conversion."
		exit 1
	fi
	cp $OUTPUT_DATASET_FILE_HDF5_OPFLOW $OUTPUT_DATASET_FILE_HDF5
	
	echo "Removing all dataset temporary files"
	rm -f $OUTPUT_DATASET_FILE_HDF5_RAW $OUTPUT_DATASET_FILE_HDF5_SORT $OUTPUT_DATASET_FILE_HDF5_OPFLOW
	
done

echo 'All done.'


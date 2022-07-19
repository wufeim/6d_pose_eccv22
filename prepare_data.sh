#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")
DATAROOT="${ROOT}/data"
PATH_PASCAL3DP="${DATAROOT}/PASCAL3D+_release1.1/"

if [ ! -d "${DATAROOT}" ]; then
    mkdir "${DATAROOT}"
fi

# Download PASCAL3D+ dataset
if [ -d "${PATH_PASCAL3DP}" ]; then
    echo "Found Pascal3D+ dataset in ${PATH_PASCAL3DP}"
else
    echo "Download Pascal3D+ dataset in ${PATH_PASCAL3DP}"
    cd "${DATAROOT}"
    wget "ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip"
    unzip "PASCAL3D+_release1.1.zip"
    rm "PASCAL3D+_release1.1.zip"
    cd "${ROOT}"
fi
if [ ! -d "${PATH_PASCAL3DP}/Image_subsets" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NsoVXW8ngQCqTHHFSW8YYsCim9EjiXS7' -O Image_subsets.zip
    unzip Image_subsets.zip
    rm Image_subsets.zip
    mv "Image_subsets" "${PATH_PASCAL3DP}"
fi

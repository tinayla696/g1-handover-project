#!/usr/bin/env bash
set -euo pipefail

S3_URI="${S3_URI:-s3://g1-gr00t-models-380421147972-us-east-1-an/motions/HandOver7/}"
LOCAL_DIR="${LOCAL_DIR:-data/motions/HandOver7}"

mkdir -p "${LOCAL_DIR}"

if [[ -f "${LOCAL_DIR}/HandOver7_unitree_g1.csv" && -f "${LOCAL_DIR}/HandOver7.npz" ]]; then
  echo "Motion files already exist in ${LOCAL_DIR}; skipping download."
  exit 0
fi

echo "Syncing motion data from ${S3_URI} to ${LOCAL_DIR} ..."
aws s3 sync "${S3_URI}" "${LOCAL_DIR}/"

echo "Done. Downloaded motion files into ${LOCAL_DIR}."

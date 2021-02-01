#!/bin/sh

# Ensure that pip in installed
# python -m ensurepip --default-pip
# pip install -r requirements.txt

# Run the machine learning model
python ./mlp/run.py
wait

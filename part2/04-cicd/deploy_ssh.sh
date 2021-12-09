#!/bin/bash
git pull origin main
`conda activate model_serving`
pip3 install -r requirements.txt

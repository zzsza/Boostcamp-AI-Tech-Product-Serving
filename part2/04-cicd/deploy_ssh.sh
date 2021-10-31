#!/bin/bash
git pull origin main
PWD=`pwd`
source $PWD/venv/bin/activate
pip install -r requirements.txt

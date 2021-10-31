#!/bin/bash
cp -r ../../assets .
sed -i -e 's+model_path: ../../assets/mask_task/model.pth+model_path: assets/mask_task/model.pth+g' config.yaml

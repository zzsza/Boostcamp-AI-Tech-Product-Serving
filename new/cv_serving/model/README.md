# CV 기초대회 베이스라인 코드

## Project Structure

```
${PROJECT}
├── dataset.py
├── inference.py
├── loss.py
├── model.py
├── README.md
├── requirements.txt
├── sample_submission.ipynb
└── train.py
```

- dataset.py : This file contains dataset class for model training and validation
- inference.py : This file used for predict the model
- loss.py : This file defines the loss functions used during training
- model.py : This file defines the model
- README.md
- requirements.txt : contains the necessary packages to be installed
- sample_submission.ipynb : an example notebook for submission
- train.py : This file used for training the model

## Getting Started

### Stteing up Vitual Enviornment

1. Install `virtualenv` if you haven't yet:

```
pip install virtualenv
```

2. Create a virtual environment in the project directory

```
cd ${PROJECT}
python -m venv /path/to/venv
```

3. Activate the virtual environment

- On Windows:

```
.\venv\Scripts\activate
```

- On Unix or MacOS:

```
source venv/bin/activate
```

4. To deactivate and exit the virtual environment, simply run:

```
deactivate
```

### Install Requirements

To Insall the necessary packages liksted in `requirements.txt`, run the following command while your virtual environment is activated:


```
pip install -r requirements.txt
```

### Usage

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.

```
SM_CHANNEL_TRAIN=/path/to/images SM_MODEL_DIR=/path/to/model python train.py
```

or 

```
python train.py --data_dir /path/to/images --model_dir /path/to/model
```

#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.

```
SM_CHANNEL_EVAL=/path/to/images SM_CHANNEL_MODEL=/path/to/model SM_OUTPUT_DATA_DIR=/path/to/output python inference.py
```

or 

```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model
```
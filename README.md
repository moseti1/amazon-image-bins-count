
# Analyzing Amazon Inventory Stock in Image Bins

 This is a `pytorch` deep learning project on [Amazon Imagery Dataset](https://registry.opendata.aws/amazon-bin-imagery/). The project utilizes pytorch to train and test on the model. 

## Project Set Up and Installation

The project uses *pytorch*, *torch* and *torchvision* packages. You can install some of the required packages using [python package manager](https://pypi.org/project/pip/) as below.

```
!pip install --no-cache-dir smdebug torch pytorch torchvision tqdm split-folders
```

## Dataset


### Overview

The dataset used in this project is that of [Amazon Image Data](https://registry.opendata.aws/amazon-bin-imagery/). According to the website's official description; 
  ***The Amazon Bin Image Dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations.***
  
  This project just uses somes of the dataset. An approximate of 10,000 images.
  
  
### Access

The data is downloaded from the specified dataset with the script below. 

```
def download_and_arrange_data():
    s3_client = boto3.client('s3')

    with open('file_list.json', 'r') as f:
        d=json.load(f)

    for k, v in d.items():
        print(f"Downloading Images with {k} objects")
        directory=os.path.join('train_data', k)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for file_path in tqdm(v):
            file_name=os.path.basename(file_path).split('.')[0]+'.jpg'
            s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),
                             os.path.join(directory, file_name))

```

The function above uses `boto3` packages to gain access the bucket specified by the data source and download the data.

To upload the data to our AWS s3 bucket we use `!aws s3 sync train_data s3://amzn-buckett/` where `amzn-buckett` is the name of the our S3 bucket we to upload the downloaded data.


## Model Training

Out model training job is achieved by using `train.py` script. The script contains functions for data loading, training testing and the main function that runs the whole project by initialiazing and calling the specified functions. We specify `entry_point` parameter in our estimator and pass the name of the name of the preffered script.Below is how the script appears.

```
estimator = PyTorch(
    entry_point="train.py",
    base_job_name='job-amazon-bins',
    role=role,
    framework_version="1.4.0",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    py_version='py3'
)
```
In the code block above is where we configure our model estimator. In it we can specify the python version to use, machine type, number of machines, name for out training job. In it we must also specify the parameter to allow access to the `sagemaker` in AWS platform.

From there we do `estimator.fit({"training": "s3://amzn-buckett/"}, wait=True)` to fit out define estimator.


## Machine Learning Pipeline

Out ML pipeline involve approximately 4 stages, namely.

 - Data .
 
   This stages involves many tasks such us downloading data from the aws data source, spliting the data for train, test and validation and uploading the data to our S3 bucket.
 
 - Training. We use the estimator created in the sagemaker notebook to accomplish this. The sagemaker runs the `train.py` script that specifies how  to load the data, train and make prediction using the fitted model.The model creation is achieved by implementing the model architecture specifications in the `train.py` script.
 
 - Model Evalution and Testing. This task involves accessing how the model performs. It checks whether the model is working as expected. This is accomplished by specifiying a loss function, from which the results are used to calculate the model accuracy.
 
 - Model Saving. Afer doing all the above steps, the model is saved into a specified path, ready for deployment and consumption.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

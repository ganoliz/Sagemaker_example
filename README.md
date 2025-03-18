# Sagemaker_example

![image](https://github.com/ganoliz/Sagemaker_example/blob/main/AWS_SageMaker.png)

Some example to familiar with AWS Sagemaker AI.

## Data Prepare and processing
In Sagemaker AI Studio we have a default bucket at S3 for storage (datasets, prediction, logs).
```python
raw_bucket = sess.default_bucket()
```

When we want to do data transform, we can use sagemaker ai build-in algorithm and assign a instance to process data.

Create our processor:
```python
sklearn_processor = SKLearnProcessor(framework_version='0.20.0',role=role,
                                    instance_type='ml.c4.xlarge', instance_count=1)
```

Prepare evaluate.py script and start a processing job:
```python
sklearn_processor.run(code=codeupload, 
                        inputs=[ProcessingInput(source=data_loc, destination='path/to/input')],
                        output=[ProcessingOutput(output_name=, source=, destination=)],
                        arguments=['--train-test-split-ratio', '0.2'])
```

## Experiment
Create an experiment
```python
cc_experiment = Experiment.create(experiment_name='train-deploy-',
                                  description='...', sagemaker_boto_client=sm,) # sm = boto3.Session().client('sagemaker')
```

Create tracking task to track parameters
```python
with Tracker.create(display_name='Preprocessing', sagemaker_boto_client=sm) as tracker:
    tracker.log_parameters({        "train_test_split_ratio": 0.2, "random_state":0 })
    # we can log the s3 uri to the dataset we just uploaded
    tracker.log_input(name="ccdefault-raw-dataset", media_type="s3/uri", value=raw_data_location)
```

Create trail
```python
cc_trial = Trial.create(trial_name=, experiment_name=cc_experiment.experiment_name,
        sagemaker_boto_client=sm)
```

## Training
We can train our model by use train.py script and run on another GPU instance

```python
# Configure the training job
estimator = Pytorch(entry_point='train.py', source_dir='code/', role=role,
                    frame_work_version='1.8.1', py_version='py3',
                    instance_count=1, instance_type='ml.p3.2xlarge', 
                    hyperparameters={'batch-size': 32, 'epochs': 10, 'learning-rate': 0.001})
data_channels = {
    'train': f'{s3_data_path}/train2017',
    'val': f'{s3_data_path}/val2017',
    'annotations': f'{s3_data_path}/annotations'
}
estimator.fit(data_channels, wait=False)
```

## Evaluation
```Python
model_data = 's3://your-bucket/path/to/model.tar.gz'
sagemaker_model = PyTorchModel(
    model_data=model_data,
    role=sagemaker.get_execution_role(),
    framework_version='1.8.0',
    py_version='py3',
    entry_point='inference.py' # Your inference script
)
transformer = sagemaker_model.transformer(instance_count=1, instance_type='ml.p3.2xlarge',
                                            output_path='path/to/output')
transformer.transform("s3://your-bucker/coco-validation-data/")
```

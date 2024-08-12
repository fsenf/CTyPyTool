# Developer Docs: What happens during Setup?

## Initiating the `cloud_classifier` class

During setup, a `cloud_classifier` class is initiated via:

```python
cc = cloud_classifier.cloud_classifier()
```


This triggers two actions:

1. the `cloud_project` class is initiated
2. the `cloud_trainer` class is initiated


## Initiating the `cloud_project` class 

The init of the cloud project handles:

1. setting the `project_path` 
2. initiating a `parameter_handler` class 
3. retrieves the default parameters from the `parameter_handler`
4. initializes an index set for masking


### Initiating the `parameter_handler` class

This sets default parameters:
- it defines json config files for general setting aspects
- it defines filelists again as json files, e.g. input files, label files, etc.
- it defines parameters for the machine learning method + details on the input and label characteristics

## Initiating the `cloud_trainer` class

The `cloud_trainer` class is responsible for 

1. training an ML model based on input and label data

2. applying a trained ML model to new input data

During initialization no further action is taken.
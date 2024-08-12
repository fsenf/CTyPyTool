# Developer Docs: What happens if you create a new project?

## Creating a new project

After `cloud_classifier` class is initialized -  see [Developer Docs: What happens during Setup?](devdocs-setup.md) - a new project can be created via

```python
cc.create_new_project(name="NewRandomForestClassifier", path="../classifiers")
```

The arguments have the following meaning:

```python
help( cc.create_new_project )
```

```
Help on method create_new_project in module cloud_project:

create_new_project(name, path=None) method of cloud_classifier.cloud_classifier instance
    Creates a persistant classifier project.
    
    
    Parameters
    ----------
    name : string
        Name of the the project that will be created
    
    path : string, optional
        Path to the directory where the project will be stored. If none is
        given, the current working directory will be used.

```

Three steps happen:
1. a new folder with name `name` is created under `path` 
2. the `parameter_handle` class is used to initialize the new project settings
3. the new project is finally loaded

### Initial new project settings via the `parameter_handler` class

This method is using a template mechanism. The template needs to be provided from somewhere. 

In the standard setting, the template should be stored under the subfolder `<package_path>/defaults`. The variable `parameter_handler.__default_path` holds the information on the template path. The currently used content is:

```bash
> tree defaults/
defaults/
├── filelists
│   ├── evaluation_sets.json
│   ├── input_files.json
│   ├── label_files.json
│   └── training_sets.json
└── settings
    ├── config.json
    └── data_structure.json

2 directories, 6 files
```

The default setting files are copied over to the new project folder.


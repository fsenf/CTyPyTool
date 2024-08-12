# CTyPyTool: Cloud Typing Python Tool

This tools is intended to help weather forecasters in assessing the quality of their cloud forecasts.

**A few facts:**
* It emulates a cloud typing methodology (see https://www.nwcsaf.org/ct2) applied to Meteosat data (see https://www.eumetsat.int/meteosat-second-generation).
* It uses standard machine learning techniques, e.g. tree & random forest classifier
* It can be applied to so-called synthetic satellite data (observsation-equivalents derived from numerical forecast data).

**Schematic**

![](docs/images/ctypytools-slide.jpg)



## Installation

### On your Local Computer

#### Cloning repository
Use the following command to clone the project to your local machine.
```
$ git clone https://github.com/fsenf/CTyPyTool
```

#### Installing Dependencies:
This project comes with a Pipfile specifying all project dependencies.
When using `pipenv` first move into the project folder with:
```
$ cd cloud_classification

```
and then use the following command to install all necesarry dependencies into your virtual environment
```
$ pipenv install

```

### On the DKRZ Servers

See [here](docs/Running_Notebooks_on_DKRZ_JupyterHub.md) to get started with `CTyPyTools` on the DKRZ Super computer.


## Getting Started
There are severeal Jupyter Notebooks explaining the basic steps for training and applying the cloud classifier.

For using an already trained classifier check out [this notebook](notebooks/Application_of_a_pretrained_classifier.ipynb)

## Contributing

Your Contribution is very welcome! Yo could either contribute with:

* providing pre-trained classifiers for a specifically defined geographical region or for certain sessions
* reporting issues, missing features or bugs
* improving code

**5 Steps for source code developers**:
1. fork the repository with the `main` branch
2. branch out into a `feature-<something>` branch in you own fork
3. update source code / software parts in your fork
4. check functionality with example notebooks
5. make a pull request onto the `main` branch in the "official" repository under https://github.com/fsenf/CTyPyTool

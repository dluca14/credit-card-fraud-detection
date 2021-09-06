# Introduction
We provide the instructions for 

# Prerequisites

In order to edit and run the code, you will need the followings:
* PyCharm - preferred IDE
* Anaconda - for installing the Python environment
* Anaconda Jupyter Notebooks - as secondary IDE, and environment to show the results
Note: for instructions about how to install a Conda environment, read the section
*Setup Anaconda environment*
* Postman


## Software to install
Additionally, you will have to have installed:
* MongoDBCompass (https://www.mongodb.com/try/download/compass)

# Setup Anaconda environment

From `https://www.anaconda.com/distribution/` choose Python 3.7 version and download 64-Bit Graphical Installer (462 MB)

## Optional, change the default directory for your Anaconda

Open an Anaconda Prompt: press Windows key, type Anaconda Prompt (and open it as Administrator)

### Generate config file

`jupyter notebook --generate-config`

Edit `jupyter_notebook_config.py` in `.username\\.jupyter` folder:

Look for  c.NotebookApp.notebook_dir and edit:
`c.NotebookApp.notebook_dir = <path_to_your_working_directory>`

Example:
`c.NotebookApp.notebook_dir = C:\\workspace`

## Install environment

In Anaconda prompt, run:

`conda create -n endava_poc python=3.7 numpy lightgbm matplotlib pillow  jupyter seaborn pymongo`

## Activate environment
`conda activate endava_poc`


## Install using pip install
pip install requirements.txt

## List packages installed in an environment

conda list -n endava_poc

## Deactivate current environment

conda deactivate endava_poc

# Python modules installation

A requirements.txt file is provided. Activate your Conda environment with:
`conda activate endava_poc` and run `pip install` in the Anaconda prompt.

For configuration of Python environment, follow the instructions below:

`pip install requirements.txt`


# Initialize Postman

Load in Postman the collection with Credit Card Fraud Detection data

# Initialize storage

You will have to initialize the storage: MongoDB database and collection


## Initialize MongoDB

Open Compass (MongoDB client) and:
 * Create a database with your username. Ex: gpreda
 * Create a collection with the name `credit_card`.
 Go in the code to storage/mongo_db.py and set:
 
```
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["gpreda"] # set your own database, on your local MongoDB env
mongo_collection = mongo_db["credit_card"]
```

## Initialize the data path

Set the path for your local storage:

Example:

`root_folder = "C:\\D\\workspace\\endava_projects\\DSCapacityDevelopment\\data"`

In the `root_folder` you will need to have 1 folder `credit-card-fraud-detection` - here you will add the files to be processed

# Run application

To run the application, you can either start the 

`python app.py`

or run `inference` from `test_inference.ipynb`

# Visualize results

You can visualize the results as following:

* Execute the tests from Postman;

* Inspect the documents associated with transactions in MongoDB (using Compass) to 
see the inference value specified in MongoDB.


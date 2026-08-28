# Effect of focal gatherings on SARS-CoV-2 spread in the US

The project was written using python 3.11.6. To reproduce the full model
results, run the following:

1. Navigate to the project directory
```
cd path/to/project/directory
```

2. Set up a virtual environment in the project directory
```
python -m venv ./venv
```

3. Activate virtual environment
```
source ./venv/bin/activate
```

4. Install required python packages
```
pip install -r ./requirements.txt 
```

5. Set the PYTHONPATH environment variable of the virtual environment to the 
project directory
```
export PYTHONPATH=path/to/project/directory
```
6. Run main.py to perform all analyses and generate all figures
```
python ./main.py
```

7. When done, deactivate the virtual environment
```
deactivate
```


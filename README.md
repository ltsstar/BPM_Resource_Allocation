# BPM Resource Allocation

## Setup
- install pipenv
- run
    ```
    pipenv install
    ```


## Running
### Discover process simulation model
- Run __init__.py in src/simulator/
- For aditionall information, see: https://github.com/bpogroup/bpo-project/

### Train task processing time prediction model
- Adjust simulator model pickle file in train_prediction_model.py
- Run
    ```
    pipenv run python
    ```

### Conduct process simulation
- Change properties in run.sh
- run
  ```
    pipenv run sh run.sh
  ```

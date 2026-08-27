# Imputation Federate

OEDISI Feed-Forward Neural Network Imputation Federate for power system injection prediction and measurement imputation in HELICS co-simulations.

## Setup

Install dependencies and create a virtual environment using `uv`:

```shell
uv sync --extra dev --extra test
```

## Run

### Run Federate Server
Start the FastAPI component configuration server:

```shell
uv run imputation-server
```

### Run HELICS Simulation
Execute the standalone federate simulation loop:

```shell
uv run imputation-sim
```

### Prepare Training Data
Extract training data from OpenDSS model definitions:

```shell
uv run python src/model.py --model=ieee123 --input=opendss/
```

### Train Model
Train the deep neural network model checkpoint:

```shell
uv run python src/train.py --model=ieee123 --output=output/
```

### Testing
Run unit and integration tests:

```shell
uv run pytest
```


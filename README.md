# Make a surface more neutral, using JAX

## How to install

run in a terminal:

```bash
poetry install
```

## How to run

### Option 1

start a shell in the virtual environment, and then run commands

```bash
poetry shell
# then e.g.
pytest
jupyter lab
```

### Option 2

start all commands with `poetry run`

```bash
poetry run pytest
poetry run jupyter lab
```

## How to add packages

Run `poetry add package-name`. Note that it can only install pip available
packages.
If binary packages are needed, we either install them by hand, or I (Romain)
can start to build containers (apptainer, docker).

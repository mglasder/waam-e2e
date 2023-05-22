# Hornets Python Template

This is the template GitHub project for our group.

## Getting started

1. Use this template:
	- **New project**: select *Use this template > Create a new repository*.
	- **Existing project**: clone this repository and copy all the files of this repository, except the `README.md`, to your own project. If you already have existing files with the same name, check what you want to take over.
1. In `conda.yml` and change `name` on line 1. For example, `myproject39`, where `39` is the Python version.
1. In `pyproject.toml`:
    - under `[project]` the `name` to your project name without the python version.
    - `authors`: change the placeholder `Your Name` to your own name and change the email address.
1. Then run the following commands to setup your environment
    1. Create a new virtual environment according to the `conda.yml` file.
        - `conda env create -f conda.yml`
    1. Activate your newly created project. Make sure to replace `<myproject>` with the project name you set in the `conda.yml`
        - `conda activate <myproject>`
    1. Install all the dependencies listed in the `pyproject.toml`
        - `hatch env create`
    1. Set up pre-commits, which correct the formatting of your files among other things.
        - `pre-commit install`

## How to install a new python module

To install a new python package, for example `hat-creaform`:

1. Add it as a dependency inside the `pyproject.toml`:

    ```diff
    [project]
    ...
    dependencies = [
        "hat-core",
    +   "hat-creaform"
    ]
    ...
    ```

2. Reload the environment to automatically install the added packages of the `pyproject.toml`:
    - `hatch env create`

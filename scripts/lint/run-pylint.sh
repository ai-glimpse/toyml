#!/bin/bash

if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run pylint ./ --rcfile=pyproject.toml
else
    pylint ./ --rcfile=pyproject.toml
fi

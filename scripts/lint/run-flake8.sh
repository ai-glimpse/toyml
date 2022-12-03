#!/bin/bash
ls
if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run flake8 ./toyml
else
    flake8 ./toyml
fi

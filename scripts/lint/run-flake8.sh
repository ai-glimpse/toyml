#!/bin/bash
ls
if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run flake8 .
else
    flake8 .
fi

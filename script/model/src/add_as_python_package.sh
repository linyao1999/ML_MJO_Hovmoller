#!/bin/bash

find . -type d -exec touch {}/__init__.py \;

#!/bin/bash


echo "all files under assets/logs, assets/models, assets/exp"

read choice


rm -r assets/logs/raw
rm -r assets/models/raw
rm -r assets/exp/raw
rm assets/logs/*.*
rm assets/models/*.*
rm assets/exp/*.*


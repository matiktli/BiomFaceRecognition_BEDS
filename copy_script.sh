#!/bin/sh
echo "Provide size:"
read sizeP
echo "Provide name:"
read newN
cp -r data_sorted_$sizeP\_$sizeP saved/$newN
cp face-data-$sizeP\-$sizeP.csv saved/$newN/

rm -rf data_sorted_$sizeP\_$sizeP

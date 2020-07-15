#! /bin/bash

if [ `lscpu | grep -i Architecture | cut -d: -f2 | cut -d' ' -f9` = "aarch64" ]
then
	DIRECTORY=/data/data/com.termux/files/usr/lib/python3.7
fi
if [ `lscpu | grep -i Architecture | cut -d: -f2 | cut -d' ' -f9` = "x86_64" ]
then
	DIRECTORY=/usr/lib/python3.6
fi

cp chem.py ${DIRECTORY}/chem.py
cp func.py ${DIRECTORY}/func.py
cp stats.py ${DIRECTORY}/stats.py
cp linear_regression.py ${DIRECTORY}/linear_regression.py

#!/bin/sh
for line in `cat requirements_latest.txt`
do
	sudo -H pip3 install $line
done

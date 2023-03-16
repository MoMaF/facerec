#! /bin/bash

container=momaf-films-20221223
suffix=films/
#swift=/scratch/project_462000189/jorma/momaf/github/facerec/python_base/bin/swift
swift=swift

if [ $# != 1 ]; then
    echo USAGE: $0 momaf-film-file-name
    exit 1
fi

f=$1

echo $0 : checking $f

if [ -e $f ]; then
    echo $0 : $f exists
    exit 0
fi

echo $0 : $f does not exist

if [ X$OS_STORAGE_URL == X ]; then
    echo $0 : OS_STORAGE_URL not defined
    exit 1
fi

if [ X$OS_AUTH_TOKEN == X ]; then
    echo $0 : OS_AUTH_TOKEN not defined
    exit 1
fi

fb=$(basename $f)
fd=$(dirname $f)

echo $0 : dir $fd file $fb

if [ ! -d $fd ]; then
    echo $0 : creating directory $fd
    mkdir $fd
    if [ $? -ne 0 ]; then
	echo $0 : failed to create directory $fd
	exit 1
    fi
fi

cd $fd

echo $0 : in $(pwd) : $swift download -p $suffix -r $container $suffix$fb

# echo $PATH

$swift download -p $suffix -r $container $suffix$fb

if [ $? -ne 0 ]; then
    echo $0 : failed to run $swift download -p $suffix -r $container $suffix$fb
    exit 1
fi

echo $0 : download of $fb in $(pwd) successful!

exit 0


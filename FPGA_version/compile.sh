#!/bin/sh

CFILES=support.c

OFILES=`echo $CFILES | sed s/\.c/\.o/g`

for i in $CFILES
do
   gcc -I$ACT_HOME/include -fPIC -c $i
done


$ACT_HOME/scripts/linkso support.so $OFILES
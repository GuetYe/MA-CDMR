#!/bin/bash

cd ryu

LOGDIR="./Log/"
if [ ! -d $LOGDIR ]
then
	mkdir $LOGDIR
fi
logfile="--log-file $LOGDIR$(date '+%Y-%m-%d-%H-%M').log"
echo "$logfile"
verbose=""
loglevel=""

for i in "$@"
do

	if [ "$i" = "verbose" ]
	then
		verbose="--verbose"
	fi
		
	if [ "$i" = "nolog" ]
	then 
		logfile=""
	fi

	if [ "$i" = "warning" ]
	then 
		loglevel='--default-log-level 30'
	fi
done

ryu-manager network_structure.py network_monitor.py network_delay.py network_shortest_path.py  arp_handler.py --observe-links $verbose $logfile $loglevel
#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 3008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 4008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 5008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 6008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 10008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 12008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13008 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14008 -1 >/dev/null 2>&1 &
sleep 0.3
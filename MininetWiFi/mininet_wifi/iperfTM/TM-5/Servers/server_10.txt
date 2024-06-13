#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 3010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 4010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 5010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 6010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 8010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 12010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13010 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14010 -1 >/dev/null 2>&1 &
sleep 0.3
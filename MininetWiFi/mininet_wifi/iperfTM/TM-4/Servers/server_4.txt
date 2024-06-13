#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 3004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 5004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 6004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 8004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 10004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 12004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13004 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14004 -1 >/dev/null 2>&1 &
sleep 0.3
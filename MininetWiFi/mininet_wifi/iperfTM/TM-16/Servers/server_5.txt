#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 3005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 4005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 6005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 8005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 10005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 12005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13005 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14005 -1 >/dev/null 2>&1 &
sleep 0.3
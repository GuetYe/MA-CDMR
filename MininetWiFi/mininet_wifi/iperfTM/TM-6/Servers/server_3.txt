#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 4003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 5003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 6003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 8003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 10003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 12003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13003 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14003 -1 >/dev/null 2>&1 &
sleep 0.3
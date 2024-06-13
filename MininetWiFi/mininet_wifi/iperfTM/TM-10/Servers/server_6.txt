#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 3006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 4006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 5006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 8006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 10006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 12006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13006 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14006 -1 >/dev/null 2>&1 &
sleep 0.3
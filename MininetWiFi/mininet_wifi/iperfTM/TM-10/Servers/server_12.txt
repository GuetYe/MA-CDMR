#!/bin/bash 
echo Initializing server listening...
iperf3 -s -p 1012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 2012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 3012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 4012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 5012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 6012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 7012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 8012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 9012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 10012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 11012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 13012 -1 >/dev/null 2>&1 &
sleep 0.3
iperf3 -s -p 14012 -1 >/dev/null 2>&1 &
sleep 0.3
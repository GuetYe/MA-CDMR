#!/bin/bash

cd mininet_wifi

echo 'root' | sudo -S mn --wifi -c
echo "loading nodes topo..."
# sudo python3 generate_topo.py
sudo python3 generate_nodes_topo.py 
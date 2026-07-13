#!/bin/bash
# Single entry point for the proxy-network sanity-check experiment.
# Safe to re-run: wipes experiments/proxy_net/output/ first.
set -e
cd "$(dirname "$0")"
rm -rf output
conda run -n uniphy python proxy_net.py

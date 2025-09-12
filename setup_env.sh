#!/bin/bash


SERVER_HOST="ec2-18-219-83-58.us-east-2.compute.amazonaws.com"

# Remove any trailing slash
SERVER_HOST="${SERVER_HOST%/}"

# Set up environment variables for WebArena websites
export SHOPPING="http://${SERVER_HOST}:7770"
export SHOPPING_ADMIN="http://${SERVER_HOST}:7780/admin"
export REDDIT="http://${SERVER_HOST}:9999"
export GITLAB="http://${SERVER_HOST}:8023"
export MAP="http://${SERVER_HOST}:3000"
export WIKIPEDIA="http://${SERVER_HOST}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://${HOSTNAME}:4399"

echo "WebArena environment variables set for server: ${SERVER_HOST}"
echo ""
echo "Environment variables:"
echo "  SHOPPING=${SHOPPING}"
echo "  SHOPPING_ADMIN=${SHOPPING_ADMIN}"
echo "  REDDIT=${REDDIT}"
echo "  GITLAB=${GITLAB}"
echo "  MAP=${MAP}"
echo "  WIKIPEDIA=${WIKIPEDIA}"
echo "  HOMEPAGE=${HOMEPAGE}"
echo ""
echo "You can now run WebArena scripts and evaluations."


export OPENAI_API_KEY="sk-71d5beabdd8849f499d591699fb4fd47"
export BASE_URL="https://api.deepseek.com/v1"


export PYTHONPATH=$PWD:$PYTHONPATH
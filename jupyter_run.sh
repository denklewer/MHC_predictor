IP_ADDRESS="$(ip a s eth0 | awk '/inet / {print$2}' | cut -d/ -f1)"

jupyter lab --ip $IP_ADDRESS --port 8888 --no-browser
# password is in /home/denklewer/.jupyter/config files  


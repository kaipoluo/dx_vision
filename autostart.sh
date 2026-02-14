sleep 5
cd ~/桌面/dx_vision/
screen \
    -S dx_vision \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    -d \
    -m \
    bash -c "./watchdog.sh"

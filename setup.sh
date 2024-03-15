sudo ifconfig wlan0 up

sudo nexutil -Iwlan0 -s500 -b -l34 -v m+IBEQGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==

sudo iw dev wlan0 interface add mon0 type monitor

sudo ip link set mon0 up

echo "You're good to go!"

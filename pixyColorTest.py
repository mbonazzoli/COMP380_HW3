import ev3dev.ev3 as ev3

# Connect Pixy camera
pixy = ev3.Sensor(address='in2')
assert pixy.connected, "Connecting PixyCam"

# Set mode
pixy.mode = 'ALL'

while True:
	print(pixy.value(0))
import SturdyRobot
import ev3dev.ev3 as ev3
import time


class honeybeeBehavior(object):

    def __init__(self, robot = None):
        self.r = robot
        self.FSM = {'seeking': self.updateSeeking,
                    'found': self.updateFound,
                    'left': self.updateLeft,
                    'right': self.updateRight,
                    'ninety': self.update90}
        self.state = 'seeking'
        self.maxLight = 0
        self.pixy = ev3.Sensor(address = "in2")
        assert self.pixy.connected, "Error while connecting Pixy camera to port"

    def updateSeeking(self):
        print("seeking")
        print(self.r.leftTouch.is_pressed)
        print(self.r.readDist())
        if self.pixy.value(0) == 5:
            return 'found'
        elif (self.r.readDist() <= 4):
            return 'ninety'
        elif self.r.leftTouch.is_pressed == 1: 
            return 'right'
        else:
            return 'left'

    def updateLeft(self):
        print("left")
        print(self.r.readDist())
        self.r.curve(.1, .2)
        if self.r.readColor == 3:
            return 'found'
        else: 
            return 'seeking'

    def updateRight(self):
        print(self.r.readDist())
        print("right")
        self.r.curve(.2, .1)
        if self.r.readColor == 3:
            return 'found'
        else: 
            return 'seeking'

    def update90(self):
        print("ninety")
        self.r.stop()
        self.r.pointerTo(90)
        self.r.mmot.wait_until_not_moving()
        if self.pixy.value(0) == 5:
            return 'found'
        else:
            # self.r.stop()
            self.r.zeroPointer()
            self.r.mmot.wait_until_not_moving()
            print('reached pointer to 0')
            # self.r.mmot.wait_until_not_moving()
            self.r.backward(.2, time = .5)
            self.r.curve(.2, -0.05, time = 1)
            # self.r.stop()
            # if self.r.readColor == 3:
            return 'seeking'
            # else:
            #     # self.r.mmot.wait_until_not_moving() 
            #     # self.r.stop()
            #     # self.r.zeroPointer(0)
            #     return 'seeking'

    def updateFound(self):
        print("Found it")
        ev3.Sound.beep()
        ev3.Sound.beep()
        ev3.Sound.beep()
        self.r.stop()
        return 'end'

    # def pixyColor(self):


    def run(self):
        """Updates the FSM by reading sensor data, then choosing based on the state"""
        updateFunc = self.FSM[self.state]
        newState = updateFunc()
        if newState is not None:
            self.state = newState
        

def runBehavior(behavObj, runTime = None):
    """Takes in a behavior object and an optional time to run. It runs
    a loop that calls the run method of the behavObj over and over until
    either the time runs out or a button is pressed."""
    buttons = ev3.Button()
    startTime = time.time()
    elapsedTime = time.time() - startTime
    ev3.Sound.speak("Starting")
    while (not buttons.any()) and ((runTime is None) or (elapsedTime < runTime)):
        behavObj.run()
        # Could add time.sleep here if need to slow loop down
        elapsedTime = time.time() - startTime
    # self.r.zeroPointer()
    # self.r.mmot.wait_until_not_moving()
    ev3.Sound.speak("Done")


if __name__ == '__main__':
    #box roby config
    config={SturdyRobot.SturdyRobot.LEFT_MOTOR: 'outC',
            SturdyRobot.SturdyRobot.RIGHT_MOTOR: 'outA',
            SturdyRobot.SturdyRobot.SERVO_MOTOR: 'outD',
            # SturdyRobot.SturdyRobot.COLOR_SENSOR: 'in2',
            # SturdyRobot.SturdyRobot.GYRO_SENSOR: 'in1',
            SturdyRobot.SturdyRobot.LEFT_TOUCH: 'in4',
            SturdyRobot.SturdyRobot.ULTRA_SENSOR: 'in3'} 

    roby = SturdyRobot.SturdyRobot('roby', config)


    buzzRoby = honeybeeBehavior(roby)

    print("Run Behavior")
    runBehavior(buzzRoby)
    print("behavior ran")


    # add code to stop robot motors
    roby.stop()
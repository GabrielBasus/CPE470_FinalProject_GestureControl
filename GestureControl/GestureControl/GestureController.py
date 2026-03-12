from time import time

import rclpy
from rclpy.node import Node
import Jetson.GPIO as GPIO

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

GAP = 0.02
N_AVG = 5
GESTURE = {"Forward": "Thumb_Up", "Backwards": "Thumb_Down", "Idle": "Open_Palm", "Following": "Pointing_Up"}

# Some common note frequencies (Hz)
NOTES = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99, "A5": 880.00,
    "REST": 0.0
}

# Example tunes as (note_name, duration_seconds)
TUNES = {
    "success": [("C5", 0.12), ("E5", 0.12), ("G5", 0.18)],
    "error":   [("G4", 0.20), ("REST", 0.05), ("G4", 0.20)],
    "startup": [("C4", 0.12), ("D4", 0.12), ("E4", 0.12), ("G4", 0.20)],
}

class GestureController(Node):

	def __init__(self):
		super().__init__('gesture_controller')

		# Create publisher and subscriber
		self.publisher_ = self.create_publisher(TwistStamped, 'gobilda/cmd_vel', 10)

		self.subscriber_ = self.create_subscription(
			LaserScan,
			'scan',
			self.lidar_callback,
			10)
		self.scan = None

		self.subscriber_ = self.create_subscription(
			String,
			'gesture',
			self.gesture_callback,
			10)
		self.gesture = None

		self.tmr = self.create_timer(1, self.timer_callback)
		self.tmr.cancel()  # Start with timer stopped

		self.i = 0
		self.total = 0.0
		self.state = "Idle"
		
		self.DistanceThreshold = 0.4  # meters
		self.command_processing = False

		#set up GPIO for buzzer
		GPIO.setmode(GPIO.BOARD)
		output_pin = 7 
		GPIO.setup(output_pin, GPIO.OUT)
		self.p = GPIO.PWM(output_pin, 440)
		self.p.start(0)
		self.play_sequence("startup")

	def timer_callback(self):
		self.state = "Idle"
		self.tmr.cancel()  # Stop the timer until the next gesture is received

	def lidar_callback(self, msg):
		self.scan = msg
		ranges = self.scan.ranges
		danger_zone = [r for r in ranges if r <= self.DistanceThreshold]
		if danger_zone:
			self.last_state = self.state
			self.state = "Recover"
			self.get_logger().info('Obstacle detected within %f meters, stopping robot after recovery' % self.DistanceThreshold)

		self.loop()

	def gesture_callback(self, msg):
		gesture_name = msg.data.split(' ')[0].strip()
		if gesture_name == GESTURE["Idle"]:
			self.state = "Idle"
			self.get_logger().info('State set to Idle')
			return

		if not self.command_processing:
			self.command_processing = True
			self.gesture = msg
			self.get_logger().info('Gesture being processed: "%s"' % msg.data)

		if gesture_name == GESTURE["Forward"]:
			self.state = "Forward"
		elif gesture_name == GESTURE["Backwards"]:
			self.state = "Backwards"
		elif gesture_name == GESTURE["Idle"]:
			self.state = "Idle"
			self.get_logger().info('State set to Idle')
		elif gesture_name == GESTURE["Following"]:
			self.state = "Following"
		else:
			self.get_logger().info('Unknown gesture: "%s"' % msg.data)

	def loop(self):
		msg = TwistStamped()
		if self.state == "Idle":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = 0.0
			self.get_logger().info('State: Idle, stopping robot')
			self.command_processing = False
			
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Forward":
			msg.twist.linear.x = 0.1
			self.get_logger().info('State: Forward, moving robot forward')
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Backwards":
			msg.twist.linear.x = -0.1
			self.get_logger().info('State: Backwards, moving robot backward')
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Following":
			if self.scan is None:
				return
			ranges = self.scan.ranges
			center_index = 540
			valid_ranges = [r for r in ranges if r > 0.1 and r <= .8]
			if not valid_ranges:
				return
			centroid_index = int(sum(i for i, r in enumerate(ranges) if r > 0.1 and r <= .8) / len(valid_ranges))
			self.get_logger().info('Centroid Index: "%s"' % centroid_index)
			distance_ahead = ranges[centroid_index]
			if ranges[centroid_index] == float('Inf'):
				msg.twist.linear.x = 0.0
				return
			dist_error = distance_ahead - self.DistanceThreshold
			if distance_ahead < self.DistanceThreshold:
				msg.twist.linear.x = min(-0.1, .5*dist_error)  # Move backward
				self.get_logger().info('moving back')
			elif distance_ahead > self.DistanceThreshold:
				msg.twist.linear.x = min(0.1, 0.5 * dist_error)  # Move forward
				self.get_logger().info('moving forward')
			else:
				msg.twist.linear.x = 0.0   # Stay
			error = centroid_index - center_index
			msg.twist.angular.z = 0.005 * error  # Proportional control
			self.get_logger().info('Turning: "%s"' % .005*error)
		elif self.state == "Recover":
			if self.last_state == "Forward":
				msg.twist.linear.x = -0.1
			elif self.last_state == "Backwards":
				msg.twist.linear.x = 0.1
			
			self.publisher_.publish(msg)
			time.sleep(0.25)  # Move for a short duration to recover

			msg.twist.linear.x = 0.0

		self.publisher_.publish(msg)
		
	def play_sequence(self, tune_name):
		seq = TUNES[tune_name]
		for note_name, dur in seq:
			freq = NOTES[note_name]
			
			if note_name == "REST":
				self.p.ChangeDutyCycle(0)
				time.sleep(dur)
			else:
				self.p.ChangeFrequency(freq)
				self.p.ChangeDutyCycle(50)
				time.sleep(dur)
				self.p.ChangeDutyCycle(0)

			time.sleep(GAP)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = GestureController()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    minimal_publisher.p.stop()
    GPIO.cleanup()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

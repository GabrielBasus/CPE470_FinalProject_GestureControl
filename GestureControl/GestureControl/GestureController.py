import time

import rclpy
from rclpy.node import Node
import Jetson.GPIO as GPIO

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32

#GPIO.cleanup(33)

GAP = 0.02
N_AVG = 5
GESTURE = {"Forward": 6, "Backwards": 5, "Idle": 0, "Following": 2, "Left": 3, "Right": 4, "Speed_1": 7, "Speed_2": 8, "Spin": 1}

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

		self.scan_subscriber_ = self.create_subscription(
			LaserScan,
			'scan',
			self.lidar_callback,
			10)
		self.scan = None

		self.subscriber_ = self.create_subscription(
			Int32,
			'gesture',
			self.gesture_callback,
			10)
		self.gesture = None
		self.tune = "none"

		self.tmr = self.create_timer(1.0, self.timer_callback)
		self.spin_tmr = self.create_timer(2.0, self.spin_callback)
		self.tune_tmr = self.create_timer(.2, self.play_sequence)

		self.i = 0
		self.j = 0
		self.k = 0
		self.total = 0.0
		self.state = "Idle"
		
		self.DistanceThreshold = 0.4  # meters
		self.command_processing = False
		self.speed = 0.25

		#set up GPIO for buzzer
		# GPIO.setmode(GPIO.BOARD)
		# output_pin = 33 
		# GPIO.setup(output_pin, GPIO.OUT)
		# self.p = GPIO.PWM(output_pin, 440)
		# self.p.start(0)
		# self.tune = "startup"

	def spin_callback(self):
		self.get_logger().info("spin_callback triggered")
		if self.state == "Spin":
			
			if self.j < 4:
				self.j+=1
			else:
				self.state = "Idle"
				self.j=0 

	def timer_callback(self):
		self.get_logger().info("timer_callback triggered")
		if self.state == "Spin":
			return
		elif self.state == "Forward" or self.state == "Backwards" or self.state == "Right" or self.state == "Left":
			self.state = "Idle"
		return
		

	def lidar_callback(self, msg):
		self.scan = msg
		ranges = self.scan.ranges
		danger_zone = [r for r in ranges if r <= self.DistanceThreshold-0.1]
		if danger_zone:
			self.last_state = self.state
			self.state = "Recover"
			self.get_logger().info('Obstacle detected within %f meters, stopping robot after recovery' % self.DistanceThreshold)
		self.loop()

	def gesture_callback(self, msg):
		gesture_cmd = msg.data
		if gesture_cmd == GESTURE["Idle"]:
			self.state = "Idle"
			self.get_logger().info('State set to Idle')
			return
		# elif gesture_cmd == GESTURE["Spin"]:
		# 	self.state = "Spin"
		# 	return

		if not self.command_processing:
			self.command_processing = True
			self.gesture = msg
			self.get_logger().info('Gesture being processed: "%s"' % msg.data)

		if gesture_cmd == GESTURE["Forward"]:
			self.state = "Forward"
			self.tmr.reset()
		elif gesture_cmd == GESTURE["Backwards"]:
			self.state = "Backwards"
			self.tmr.reset()
		elif gesture_cmd == GESTURE["Idle"]:
			self.state = "Idle"
		elif gesture_cmd == GESTURE["Following"]:
			self.state = "Following"
			self.tmr.reset()
		elif gesture_cmd == GESTURE["Left"]:
			self.state = "Left"
			self.tmr.reset()
		elif gesture_cmd == GESTURE["Right"]:
			self.state = "Right"
			self.tmr.reset()
		elif gesture_cmd == GESTURE["Speed_1"]:
			self.speed = 0.25
		elif gesture_cmd == GESTURE["Speed_2"]:
			self.speed = 0.5
		elif gesture_cmd == GESTURE["Spin"]:
			self.state = "Spin"
			self.spin_tmr.reset()
		else:
			self.get_logger().info('Unknown gesture: "%s"' % msg.data)

		self.tmr.reset()

	def loop(self):
		msg = TwistStamped()
		if self.state == "Idle":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = 0.0
			self.get_logger().info('State: Idle, stopping robot')
			self.command_processing = False
			self.publisher_.publish(msg)
			#self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Forward":
			msg.twist.linear.x = self.speed
			self.publisher_.publish(msg)
			self.get_logger().info('State: Forward, moving robot forward')
			#self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Backwards":
			msg.twist.linear.x = -self.speed
			self.publisher_.publish(msg)
			self.get_logger().info('State: Backwards, moving robot backward')
			#self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Left":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = -0.5
			self.publisher_.publish(msg)
			self.get_logger().info('State: Left, turning robot left')
			#self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Right":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = 0.5
			self.publisher_.publish(msg)
			self.get_logger().info('State: Right, turning robot right')
			#self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Following":
			if self.scan is None:
				return
			ranges = self.scan.ranges
			center_index = 540
			valid_ranges = [r for r in ranges if r > 0.1 and r <= 1.5]
			if not valid_ranges:
				return
			valid_indices = [i for i, r in enumerate(ranges) if r > 0.5 and r <= 1.5]
			centroid_index = int(sum(valid_indices) / len(valid_indices))
			self.get_logger().info('Centroid Index: "%s"' % centroid_index)
			distance_ahead = ranges[centroid_index]
			if ranges[centroid_index] == float('Inf'):
				msg.twist.linear.x = 0.0
				return
			dist_error = distance_ahead - self.DistanceThreshold
			if distance_ahead < self.DistanceThreshold:
				msg.twist.linear.x = min(-0.25, .5*dist_error)  # Move backward
				self.get_logger().info('moving back')
			elif distance_ahead > self.DistanceThreshold:
				msg.twist.linear.x = max(0.25, 0.5 * dist_error)  # Move forward
				self.get_logger().info('moving forward')
			else:
				msg.twist.linear.x = 0.0   # Stay
			error = centroid_index - center_index
			msg.twist.angular.z = 0.005 * error  # Proportional control
			self.get_logger().info('Turning: "%s"' % .05*error)
			self.publisher_.publish(msg)
		elif self.state == "Recover":
			# if self.last_state == "Forward":
			# 	msg.twist.linear.x = -0.25
			# elif self.last_state == "Backwards":
			# 	msg.twist.linear.x = 0.25
			
			# self.publisher_.publish(msg)
			# self.tmr.reset()
			if self.scan is None:
				return
			ranges = self.scan.ranges
			center_index = 540
			valid_ranges = [r for r in ranges if r <= .5]
			if not valid_ranges:
				return
			valid_indices = [i for i, r in enumerate(ranges) if r <= .5]
			centroid_index = int(sum(valid_indices) / len(valid_indices))
			self.get_logger().info('Centroid Index: "%s"' % centroid_index)
			distance_ahead = ranges[centroid_index]
			if ranges[centroid_index] == float('Inf'):
				msg.twist.linear.x = 0.0
				return
			dist_error = distance_ahead - self.DistanceThreshold
			if distance_ahead > self.DistanceThreshold:
				msg.twist.linear.x = min(-0.1, 0.5 * dist_error)  # Move away
				self.get_logger().info('moving forward')
			else:
				msg.twist.linear.x = 0.0   # Stay
			error = centroid_index - center_index
			msg.twist.angular.z = 0.005 * error  # Proportional control
			self.get_logger().info('Turning: "%s"' % .005*error)
			self.publisher_.publish(msg)

		elif self.state == "Spin":
			self.get_logger().info('Spinning')
			msg.twist.angular.z = .785
			self.publisher_.publish(msg)
		else:
			self.state = "Idle"

	def play_sequence(self):
		
		if self.tune == "success":
			if self.k < len(TUNES["success"]):
				freq = NOTES[TUNES["success"][self.k][0]]
				self.p.ChangeFrequency(freq)
				self.p.ChangeDutyCycle(50)
				self.k+=1
			else:
				self.k = 0
				self.p.ChangeDutyCycle(0)
				self.tune = "none"

			
		elif self.tune == "error":
			if self.k < len(TUNES["error"]):
				freq = NOTES[TUNES["error"][self.k][0]]
				self.p.ChangeFrequency(freq)
				self.p.ChangeDutyCycle(50)
				self.k+=1
			else:
				self.k = 0
				self.p.ChangeDutyCycle(0)
				self.tune = "none"

		elif self.tune == "startup":
			if self.k < len(TUNES["startup"]):
				freq = NOTES[TUNES["startup"][self.k][0]]
				self.p.ChangeFrequency(freq)
				self.p.ChangeDutyCycle(50)
				self.k+=1
			else:
				self.k = 0
				self.p.ChangeDutyCycle(0)
				self.tune = "none"

		else:
			#self.p.ChangeDutyCycle(0)
			self.k = 0

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = GestureController()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    # minimal_publisher.p.stop()
    # GPIO.cleanup(output_pin)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

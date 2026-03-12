import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32


N_AVG = 5
GESTURE = {"Forward": 6, "Backward": 5, "Idle": 0, "Following": 2, "Left": 3, "Right": 4}

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
			Int32,
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

	def timer_callback(self):
		self.state = "Idle"
		self.tmr.cancel()  # Stop the timer until the next gesture is received

	def lidar_callback(self, msg):
		self.scan = msg
		ranges = self.scan.ranges
		danger_zone = [r for r in ranges if r <= self.DistanceThreshold]
		if danger_zone:
			self.state = "Idle"
			self.get_logger().info('Obstacle detected within %f meters, stopping robot' % self.DistanceThreshold)

		self.loop()

	def gesture_callback(self, msg):
		gesture_cmd = msg.data
		if gesture_cmd == GESTURE["Idle"]:
			self.state = "Idle"
			self.get_logger().info('State set to Idle')
			return

		if not self.command_processing:
			self.command_processing = True
			self.gesture = msg
			self.get_logger().info('Gesture being processed: "%s"' % msg.data)

		if gesture_cmd == GESTURE["Forward"]:
			self.state = "Forward"
		elif gesture_cmd == GESTURE["Backward"]:
			self.state = "Backwards"
		elif gesture_cmd == GESTURE["Idle"]:
			self.state = "Idle"
			self.get_logger().info('State set to Idle')
		elif gesture_cmd == GESTURE["Following"]:
			self.state = "Following"
		elif gesture_cmd == GESTURE["Left"]:
			self.state = "Left"
		elif gesture_cmd == GESTURE["Right"]:
			self.state = "Right"
		else:
			self.get_logger().info('Unknown gesture: "%s"' % msg.data)

	def loop(self):
		msg = TwistStamped()
		if self.state == "Idle":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = 0.0
			self.get_logger().info('State: Idle, stopping robot')
			self.command_processing = False
			self.publisher_.publish(msg)
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Forward":
			msg.twist.linear.x = 0.1
			self.publisher_.publish(msg)
			self.get_logger().info('State: Forward, moving robot forward')
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Backwards":
			msg.twist.linear.x = -0.1
			self.publisher_.publish(msg)
			self.get_logger().info('State: Backwards, moving robot backward')
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Left":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = -0.2
			self.publisher_.publish(msg)
			self.get_logger().info('State: Left, turning robot left')
			self.tmr.reset()  # Reset the timer to start counting down from now
			return
		elif self.state == "Right":
			msg.twist.linear.x = 0.0
			msg.twist.angular.z = 0.4
			self.publisher_.publish(msg)
			self.get_logger().info('State: Right, turning robot right')
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
			self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = GestureController()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

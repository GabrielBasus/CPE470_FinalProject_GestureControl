import rclpy
import numpy as np
import cv2
from cv_bridge import CvBridge
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class VisionPublisher(Node):

    def __init__(self):
        super().__init__('vision_publisher')

        self.bridge = CvBridge()

        # Setup gesture recognizer
        base_options = python.BaseOptions(model_asset_path='src/GestureControl/GestureControl/gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        self.draw = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

        self.publisher_ = self.create_publisher(String, 'gesture', 10)

        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/image_raw',
            self.image_callback,
            10
        )

        self.debug_ = self.create_publisher(Image, 'debug', 10)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").reshape(msg.height, msg.width, -1)
        #frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # GestureRecognizer expects a MediaPipe Image in RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.recognizer.recognize(mp_image)

        # Draw landmarks and overlay gesture label for each detected hand
        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):

                # Convert normalized landmarks to proto for drawing
                landmark_proto = landmark_pb2.NormalizedLandmarkList()
                landmark_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ])
                self.draw.draw_landmarks(frame, landmark_proto, self.hand_connections)

                # Overlay gesture label if available
                if result.gestures and result.gestures[i]:
                    top_gesture = result.gestures[i][0]
                    label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                    msg_out = String()
                    msg_out.data = label
                    self.publisher_.publish(msg_out)  # Publish the gesture label as a string message

                    # Draw label at wrist position (landmark 0)
                    wrist = hand_landmarks[0]
                    h, w, _ = frame.shape
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, label, (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.debug_.publish(debug_msg)
        
        #cv2.imshow("Gesture Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = VisionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
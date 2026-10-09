import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from std_msgs.msg import Int32

class Color_sort(Node):
    def __init__(self):
        super().__init__('color_sort')
        self.get_logger().info("the color sort node has started....")

        self.red_count = 0
        self.green_count = 0
        self.blue_count = 0

        self.red_detected = False
        self.green_detected = False
        self.blue_detected = False
    

        # Initialize the camera webcam feed

        self.cap = cv2.VideoCapture(0)

        # Define HSV color boundaries for Red, Green, and Blue
        # Note: Red wraps around the 0-180 Hue scale, so it can have two ranges.
        self.color_ranges = {
            "Red": [
                {"lower": np.array([0, 120, 70]), "upper": np.array([10, 255, 255])},
                {"lower": np.array([170, 120, 70]), "upper": np.array([180, 255, 255])}
            ],
            "Green": [
                {"lower": np.array([36, 100, 40]), "upper": np.array([86, 255, 255])}
            ],
            "Blue": [
                {"lower": np.array([94, 80, 2], dtype=np.uint8), "upper": np.array([126, 255, 255], dtype=np.uint8)}
            ]
        }

        # Standard BGR colors for drawing the text labels and bounding boxes
        self.display_colors = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0)
        }


        self.cv_start_timer = self.create_timer(0.03,self.cvStart)

        self.red_publisher_ = self.create_publisher(Int32 , 'red_topic',10)
        self.blue_publisher_ = self.create_publisher(Int32 , 'blue_topic',10)
        self.green_publisher_ = self.create_publisher(Int32 , 'green_topic',10)

        self.red_timer = self.create_timer(1.0,self.red_)
        self.blue_timer = self.create_timer(1.0,self.blue_)
        self.green_timer = self.create_timer(1.0,self.green_)

    def red_(self):
        msg = Int32()
        msg.data = self.red_count
        self.get_logger().info(f"red -> {msg.data}")
        self.red_publisher_.publish(msg)
        
    def blue_(self):
        msg = Int32()
        msg.data = self.blue_count
        self.get_logger().info(f"blue -> {msg.data}")
        self.blue_publisher_.publish(msg)

    def green_(self):
        msg = Int32()
        msg.data = self.green_count
        self.get_logger().info(f"green -> {msg.data}")
        self.green_publisher_.publish(msg)  


    def cvStart(self):

            current_counts = {
                "Red": 0,
                "Green": 0,
                "Blue": 0
            }
        
            ret, frame = self.cap.read()
            if not ret:
                return

            # Blur the frame to eliminate high-frequency noise
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            
            # Convert BGR video frame to HSV color space
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            for color_name, ranges in self.color_ranges.items():
                # Generate the color mask
                if len(ranges) == 2:
                    # Combine two mask ranges if dealing with Red hue wrapping
                    mask1 = cv2.inRange(hsv, ranges[0]["lower"], ranges[0]["upper"])
                    mask2 = cv2.inRange(hsv, ranges[1]["lower"], ranges[1]["upper"])
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, ranges[0]["lower"], ranges[0]["upper"])

                # Perform morphological cleaning to eliminate tiny background specks
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # Detect contours inside the generated binary mask
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    # Filter out small detections to minimize false positives
                    if area > 800: 
                       

                        if color_name == "Red" and not self.red_detected:
                            self.red_count += 1
                            self.red_detected = True

                        elif color_name == "Green" and not self.green_detected:
                            self.green_count += 1
                            self.green_detected = True

                        elif color_name == "Blue" and not self.blue_detected:
                            self.blue_count += 1
                            self.blue_detected = True


                        
                        current_counts[color_name] += 1

                        # Compute the bounding rectangle dimensions
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        
                        # Draw box around identified object
                        cv2.rectangle(frame, (x, y), (x + w, y + h), self.display_colors[color_name], 2)
                        
                        # Label the object with its sorted classification
                        cv2.putText(frame, f"Sorted: {color_name}", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.display_colors[color_name], 2)
                        
            if current_counts["Red"] == 0:
                self.red_detected = False

            if current_counts["Green"] == 0:
                self.green_detected = False

            if current_counts["Blue"] == 0:
                self.blue_detected = False           
            
            
            cv2.putText(frame, f"Red: {current_counts['Red']}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Green: {current_counts['Green']}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Blue: {current_counts['Blue']}", (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 



            # cv2.putText(frame, f"Total Red: {self.red_count}", (20, 30),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # cv2.putText(frame, f"Total Green: {self.green_count}", (20, 60),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # cv2.putText(frame, f"Total Blue: {self.blue_count}", (20, 90),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)




            # Show the interactive output feed
            cv2.imshow("Real-Time Object Color Sorting", frame)

            # Exit program instantly when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):        
                self.cap.release()
                cv2.destroyAllWindows()
                return


def main(args = None):
    rclpy.init(args = args)
    node = Color_sort()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()            
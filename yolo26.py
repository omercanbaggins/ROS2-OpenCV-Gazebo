#!/home/omercan/yolo26-env/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime
import cv2
import numpy as np

class YOLO26Node(Node):
    def __init__(self):
        super().__init__('yolo26_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_cb, 10
        )
        
        # Ensure the path is correct
        model_path = "/home/omercan/ros2_ws/src/deneme125/yolo26n.onnx"
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

    def preprocess(self, img):
        # YOLO standard: 640x640, RGB, normalized
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_reshaped = np.transpose(img_normalized, (2, 0, 1))[None, ...]  # BCHW
        return img_reshaped

    def postprocess(self, output, original_size):
        # output shape: [1, 84, 8400] -> [84, 8400]
        output = output[0][0]
        output = output.transpose() # [8400, 84]

        boxes = []
        confidences = []
        class_ids = []

        orig_h, orig_w = original_size
        x_factor = orig_w / 640
        y_factor = orig_h / 640

        for row in output:
            score = row[4:].max() # Get max class probability
            if score > self.conf_threshold:
                class_id = row[4:].argmax()
                
                # YOLO output is usually [cx, cy, w, h]
                cx, cy, w, h = row[0:4]
                
                # Scale to original image pixels
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                confidences.append(float(score))
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        return indices, boxes, confidences, class_ids

    def image_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = cv_img.shape[:2]

        # 1. Inference
        inp = self.preprocess(cv_img)
        outputs = self.session.run(None, {self.input_name: inp})

        # 2. Parse Results
        indices, boxes, confs, ids = self.postprocess(outputs, (h, w))

        # 3. Draw Detections
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                label = f"ID: {ids[i]} {confs[i]:.2f}"
                
                # Draw Box
                cv2.rectangle(cv_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                # Draw Label
                cv2.putText(cv_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Show Result (for debugging)
        cv2.imshow("YOLOv11 Detection", cv_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLO26Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
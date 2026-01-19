import rclpy
from rclpy.node import Node
from ros2_fashion_interfaces.msg import FashionItem
from ultralytics import YOLO
import os

class FashionDetectionPublisher(Node):  
    def __init__(self, name):
        super().__init__(name)
        # 配置参数
        self.MODEL_PATH = '/home/rex/fashion-mnist/yolo_file/runs/detect/train6/weights/best.pt'
        self.VAL_DIR = '/home/rex/fashion-mnist/yolo_file/mnist_det/images/val'
        self.SAVE_RESULTS = False
    
        # 类别名称映射
        self.CLASS_NAMES = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", 
            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        self.get_logger().info("It's %s!" % name)
        self.command_publisher_ = self.create_publisher(FashionItem, "fashion_detection", 10)
        self.timer = self.create_timer(2.0, self.timer_callback)

    def timer_callback(self):
        model = self.load_model(self.MODEL_PATH)
        val_images = self.get_val_images(self.VAL_DIR)
        
        for image_path in val_images:
            detections = self.detect_image(model, image_path, self.SAVE_RESULTS)
            image_name = os.path.basename(image_path)
            
            for det in detections:
                class_name = self.CLASS_NAMES[det['class']]
                confidence = det['confidence']
                
                msg = FashionItem()
                msg.imgname = image_name
                msg.img = class_name
                msg.confidence = confidence
                
                self.command_publisher_.publish(msg)
                self.get_logger().info(
                    f'Published: imgname={msg.imgname}, img={msg.img}, '
                    f'confidence={msg.confidence:.4f}'
                )

    def load_model(self, model_path):
        return YOLO(model_path)
    
    def get_val_images(self, val_dir):
        return [os.path.join(val_dir, file) for file in os.listdir(val_dir) 
                if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def detect_image(self, model, image_path, save_results=False):
        results = model.predict(
            source=image_path,
            data='mnist.yaml',
            save=save_results,
            show=False,
            verbose=False
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0])
                })
        
        return detections

def main(args=None):
    rclpy.init(args=args)
    node = FashionDetectionPublisher("fashion_detection_publisher")  # ✅ 关键修正：类名匹配
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

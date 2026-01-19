import rclpy
from rclpy.node import Node
from ros2_fashion_interfaces.msg import FashionItem

class FashionDetectionSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.subscription = self.create_subscription(
            FashionItem,
            'fashion_detection',
            self.listener_callback,
            10)
        self.get_logger().info('Fashion Detection Subscriber has been started')
        # 为每张图片维护序号计数器
        self.image_counter = {}

    def listener_callback(self, msg):
        # 为当前图片初始化计数器（如果尚未存在）
        if msg.imgname not in self.image_counter:
            self.image_counter[msg.imgname] = 0
        
        # 递增序号
        self.image_counter[msg.imgname] += 1
        
        # 核心修改：在每行开头添加"Received message"
        self.get_logger().info(
            f"Received message: {msg.imgname}: {msg.img} (confidence={msg.confidence:.4f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FashionDetectionSubscriber('fashion_detection_subscriber')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
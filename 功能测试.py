from ultralytics import YOLO
import os

# 加载YOLO模型
def load_model(model_path):
    return YOLO(model_path)

# 获取val目录下的所有图像
def get_val_images(val_dir):
    return [os.path.join(val_dir, file) for file in os.listdir(val_dir) if file.endswith('.jpg')]

# 对单个图像进行推理
def detect_image(model, image_path):
    results = model.predict(
        source=image_path,
        save=False,
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

# 主函数
def main():
    # 配置参数
    MODEL_PATH = 'runs/detect/train3/weights/best.pt'
    VAL_DIR = 'mnist_det/images/val'
    
    # 加载模型
    model = load_model(MODEL_PATH)
    
    # 获取val图像列表
    val_images = get_val_images(VAL_DIR)
    
    # 遍历所有图像进行检测
    for image_path in val_images:
        detections = detect_image(model, image_path)
        
        # 输出结果
        image_name = os.path.basename(image_path)
        print(f"\n{image_name}")
        for det in detections:
            print(f"{det['class']} {det['confidence']:.4f}")

if __name__ == "__main__":
    main()

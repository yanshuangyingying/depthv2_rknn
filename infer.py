import cv2
import numpy as np
from rknnlite.api import RKNNLite
import argparse
import time

def preprocess_image(image_path, width, height):
    """
    预处理输入图像
    :param image_path: 图像文件路径
    :param width: 目标宽度
    :param height: 目标高度
    :return: 预处理后的图像
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    image = (image / 255.0)
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # 标准化
    # image = image.transpose((2, 0, 1))  # 转换为CHW格式
    image = np.expand_dims(image, axis=0)  # 增加 batch 维度
    return image.astype(np.float32)

def infer_rknn_model(image, model_path):
    """
    使用RKNN执行模型推理
    :param image: 预处理后的图像
    :param model_path: RKNN模型路径
    :return: 推理结果
    """
    # 创建RKNN对象
    rknn = RKNNLite()

    # 加载RKNN模型
    print("Loading RKNN model...")
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("Failed to load RKNN model.")
        return None

    # 初始化RKNN runtime
    print("Initializing RKNN runtime...")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print("Failed to initialize RKNN runtime.")
        return None
        print("Warming up the model...")

    # for _ in range(10):
    #     rknn.inference(inputs=[image])
    # 执行推理
    print("Running inference...")
    star=time.time()
    outputs = rknn.inference(inputs=[image])
    print(time.time()-star)
    print(np.shape(outputs))

    # 释放RKNN资源
    rknn.release()

    return outputs

def postprocess_depth(depth, orig_shape, output_path):
    """
    后处理深度图并保存
    :param depth: 推理结果
    :param orig_shape: 原始图像尺寸
    :param output_path: 保存结果的路径
    """
    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    
    # 去掉 batch 维度
    depth = depth[0].squeeze()
    
    print(f"Depth shape before resize: {depth.shape}")
    
    # 归一化到 [0, 255]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    # 确保 depth 的最小值和最大值在 [0, 255] 范围内
    depth = np.clip(depth, 0, 255)
    
    depth = depth.astype(np.uint8)
    
    print(f"Depth shape after normalization: {depth.shape}")
    print(f"Depth min: {depth.min()}, max: {depth.max()}")
    # depth = depth[:172,:172]
    depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    cv2.imwrite(output_path,depth_color)
    # 保存深度数组到文本文件
    # np.savetxt('array.txt', depth, fmt='%d', comments='')
    # print("数组已保存到 'array.txt'")
    # depth = depth[:172,:172]
    # 保存深度图像
    # im = Image.fromarray(depth)
    # im = im.resize((orig_shape[1], orig_shape[0]), Image.NEAREST)  # 如果需要调整尺寸
    
    # im.save(output_path)
    print(f"深度图像已保存到 {output_path}")

    print("*************************")

def main(img_path, model_path, output_path, width=280, height=364):
    """
    主函数
    :param img_path: 输入图像路径
    :param model_path: RKNN模型路径
    :param output_path: 保存结果的路径
    :param width: 目标宽度
    :param height: 目标高度
    """
    # 预处理图像
    image = preprocess_image(img_path, width, height)

    # 执行推理
    depth = infer_rknn_model(image, model_path)

    # 后处理并保存结果
    if depth is not None:
        orig_shape = cv2.imread(img_path).shape[:2]  # 获取原始图像尺寸
        postprocess_depth(depth, orig_shape, output_path)

'''
使用示例:
python v1infer.py --img 518.jpg --model V1.rknn --output res.jpg

python v1infer.py --img 518.jpg --model 2depth_anything_v2_vits_dynamic.rknn --output res.jpg
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKNN模型推理及深度图处理")
    parser.add_argument("--img", required=True, help="输入图像路径")
    parser.add_argument("--model", required=True, help="RKNN模型路径")
    parser.add_argument("--output", required=True, help="保存结果的路径")
    args = parser.parse_args()

    main(args.img, args.model, args.output)


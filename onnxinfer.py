import onnx
import onnxruntime as ort
import cv2
import numpy as np
import time
import argparse


def preprocess_image(image_path, width, height):
    """
    预处理输入图像
    :param image_path: 图像文件路径
    :param width: 目标宽度
    :param height: 目标高度
    :return: 预处理后的图像
    """
    image = cv2.imread(image_path)  # 读取图像
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # 转换为RGB并归一化到[0, 1]
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)  # 调整大小
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # 标准化
    image = image.transpose(2, 0, 1)[None].astype(np.float32)  # 转换为CHW格式并增加批次维度
    return image


def infer_onnx_model(image, model_path):
    """
    使用ONNX Runtime执行模型推理
    :param image: 预处理后的图像
    :param model_path: ONNX模型路径
    :return: 推理结果
    """
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("准备推理")
    results = session.run([output_name], {input_name: image})
    return results[0]


def postprocess_depth(depth, orig_shape, output_path):
    """
    后处理深度图并保存
    :param depth: 推理结果
    :param orig_shape: 原始图像尺寸
    :param output_path: 保存结果的路径
    """
    depth = depth.squeeze()  # 去掉批次维度
    depth = depth.squeeze()
    print(depth.shape)
    # depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)  # 调整到原始尺寸
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0  # 归一化到[0, 255]
    depth = depth.astype(np.uint8)
    cv2.imwrite('gray.jpg',depth)
    #保存数组
    np.savetxt('array.txt', depth, fmt='%.2f', comments='')
    print("数组已保存到 'array.txt'")
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)  # 应用调色板
    cv2.imwrite(output_path, depth_color)  # 保存结果


def main(img_path, model_path, output_path, width=140, height=140):
    """
    主函数
    :param img_path: 输入图像路径
    :param model_path: ONNX模型路径
    :param output_path: 保存结果的路径
    :param width: 目标宽度
    :param height: 目标高度
    """
    # 预处理图像
    image = preprocess_image(img_path, width, height)

    # 执行推理
    stra = time.time()
    depth = infer_onnx_model(image, model_path)
    print(time.time()-stra)

    # 后处理并保存结果
    orig_shape = cv2.imread(img_path).shape[:2]  # 获取原始图像尺寸
    postprocess_depth(depth, orig_shape, output_path)

'''
使用示例
python onnx_inference.py --img path/to/input_image.jpg --model path/to/model.onnx --output path/to/output_depth_map.jpg
python onnx_inference.py --img 2.jpg --model weights/vitb.onnx --output output_depth_map.jpg
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX模型推理及深度图处理")
    parser.add_argument("--img",default="518.jpg", help="输入图像路径")
    parser.add_argument("--model", default="model_uint8.onnx", help="ONNX模型路径")
    parser.add_argument("--output", default="2colorpic.jpg", help="保存结果的路径")
    args = parser.parse_args()

    main(args.img, args.model, args.output)

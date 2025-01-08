from paddleocr import PaddleOCR
from PIL import Image
import cv2
import base64
import io
import logging
from openai import OpenAI
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTableParser:
    def __init__(self, dashscope_api_key):
        """初始化OCR和视觉模型"""
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=False,
            show_log=False
        )
        
        # 初始化DashScope客户端
        self.client = OpenAI(
            api_key=dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _encode_pil_image(self, pil_image):
        """将PIL Image转换为base64"""
        buffered = io.BytesIO()
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def _process_ocr_results(self, ocr_result):
        """处理OCR结果，返回结构化的文本信息"""
        if not ocr_result or not ocr_result[0]:
            return []
            
        structured_results = []
        for line in ocr_result[0]:
            box = line[0]  # 坐标信息
            text = line[1][0]  # 识别的文本
            confidence = line[1][1]  # 置信度
            
            # 计算中心点
            center_x = sum(p[0] for p in box) / 4
            center_y = sum(p[1] for p in box) / 4
            
            structured_results.append({
                'text': text,
                'position': (center_x, center_y),
                'confidence': confidence,
                'box': box
            })
            
        return structured_results

    def parse_table_with_vision(self, image_path):
        """使用OCR和视觉模型解析表格"""
        try:
            # 1. 首先进行OCR识别
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("无法读取图像文件")
                
            # OCR识别
            ocr_result = self.ocr.ocr(img)
            structured_ocr = self._process_ocr_results(ocr_result)
            
            # 将OCR结果组织成提示信息
            ocr_text = "OCR识别结果如下：\n"
            for item in structured_ocr:
                ocr_text += f"- 位置({item['position'][0]:.1f}, {item['position'][1]:.1f}): {item['text']}\n"
            
            # 2. 使用视觉模型进行解析
            # 准备图像
            pil_image = Image.open(image_path)
            base64_image = self._encode_pil_image(pil_image)
            
            # 构建提示信息
            prompt = f"""这是一个表格图像。我已经使用OCR进行了初步识别，请帮我：
1. 理解表格的整体结构
2. 注意数字区间、小数点等细节的准确性
3. 注意合并单元格、跨行、跨列的情况
========OCR初步识别的结果如下：==========
{ocr_text}
==================
请以Json格式输出完整的、经过整理的表格内容。
此外，对于Json格式未能完整表达的内容和信息：请以文字增加辅助说明，并给出几个例子。"""

            # 调用视觉模型
            completion = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"表格解析错误: {str(e)}")
            raise

def main():
    # 使用示例
    import os
    # 从环境变量获取API密钥
    api_key = os.getenv("DASH_SCOPE_API_KEY","")

    parser = EnhancedTableParser(api_key)
    try:
        result = parser.parse_table_with_vision("image.png")
        print("解析结果：")
        print(result)
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image
import cv2
import base64
import io
import os
import re
import logging
from openai import OpenAI
import numpy as np
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFParser:
    def __init__(self, dashscope_api_key):
        """初始化OCR和视觉模型"""
        # 初始化PaddleOCR，复用同一个实例
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=False,
            show_log=False,
            rec=True,
            det=True
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


    def _detect_table_border(self, image):
        """检测表格外边框"""
        try:
            # 检查图像是否有效
            if image is None or image.size == 0:
                logger.error("无效的图像数据")
                return None, []
                
            # 确保图像尺寸足够大
            min_dimension = 30  # 最小尺寸要求
            if image.shape[0] < min_dimension or image.shape[1] < min_dimension:
                logger.error(f"图像尺寸过小: {image.shape}")
                return None, []

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 自适应二值化
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 根据图像尺寸动态计算核大小
            kernel_size = max(3, min(image.shape) // 100)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            
            # 使用形态学操作提取线条
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, []
            
            # 筛选可能的表格边框
            potential_tables = []
            min_area = image.shape[0] * image.shape[1] * 0.01  # 最小面积为图像面积的1%
            
            for contour in contours:
                # 获取轮廓的近似多边形
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 计算轮廓的基本特征
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # 跳过过小的区域
                if area < min_area:
                    continue
                    
                rect_area = w * h
                aspect_ratio = w / float(h)
                extent = float(area) / rect_area
                
                # 表格边框的判断条件
                if (len(approx) >= 4 and  # 至少4个角点
                    0.2 < aspect_ratio < 5 and  # 合理的宽高比
                    extent > 0.6):  # 较高的填充率
                    
                    potential_tables.append({
                        'contour': contour,
                        'rect': (x, y, w, h),
                        'area': area,
                        'score': extent  # 使用填充率作为评分
                    })
            
            # 如果没有找到符合条件的边框，返回None
            if not potential_tables:
                return None, []
            
            # 按评分排序，选择最佳候选
            best_table = max(potential_tables, key=lambda x: x['score'])
            return best_table['rect'], best_table['contour']
            
        except Exception as e:
            logger.error(f"表格边框检测失败: {str(e)}")
            return None, []

    def _analyze_internal_structure(self, image, table_rect, structured_ocr):
        """分析表格内部结构"""
        try:
            if image is None or table_rect is None:
                return False
                
            x, y, w, h = table_rect
            
            # 确保坐标有效
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return False
            if x + w > image.shape[1] or y + h > image.shape[0]:
                return False
            
            # 提取表格区域
            table_region = image[y:y+h, x:x+w]
            if table_region.size == 0:
                return False
                
            gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 动态计算核大小
            h_kernel_size = max(3, w//30)
            v_kernel_size = max(3, h//30)
            
            # 检测内部线条
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # 统计表格内的文本
            table_texts = []
            for text in structured_ocr:
                tx, ty = text['position']
                if (x < tx < x+w and y < ty < y+h):
                    table_texts.append(text)
            
            return len(table_texts) > 0
            
        except Exception as e:
            logger.error(f"分析表格内部结构失败: {str(e)}")
            return False

    def _is_likely_table(self, image, ocr_results):
        """改进的表格检测方法，主要基于边框检测"""
        # 1. 首先检测表格边框
        table_rect, table_contour = self._detect_table_border(image)
        
        if table_rect is None:
            logger.info("未检测到表格边框")
            return False
        
        # 2. 分析表格内部结构
        has_internal_structure = self._analyze_internal_structure(
            image, table_rect, ocr_results
        )
        
        if not has_internal_structure:
            logger.info("检测到边框但内部结构不符合表格特征")
            return False
        
        logger.info("检测到有效表格")
        return True

    def process_pdf_with_tables(self, pdf_path, output_dir="output", batch_size=5):
        """处理PDF文件，识别文本和表格，支持分批处理
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            batch_size: 每批处理的页数
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info("正在转换PDF为图片...")
            images = convert_from_path(
                pdf_path,
                dpi=300,
                fmt='jpg'
            )
            
            total_pages = len(images)
            logger.info(f"共检测到 {total_pages} 页")
            
            # 读取处理进度
            progress_file = os.path.join(output_dir, "progress.json")
            if os.path.exists(progress_file):
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    last_processed_page = progress.get('last_processed_page', -1)
            else:
                last_processed_page = -1
            
            # 分批处理
            for batch_start in range(last_processed_page + 1, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"处理批次: {batch_start + 1} - {batch_end}")
                
                batch_result = {
                    "document_info": {
                        "total_pages": total_pages,
                        "batch_start": batch_start + 1,
                        "batch_end": batch_end,
                        "processed_time": datetime.now().isoformat()
                    },
                    "pages": []
                }
                
                # 处理当前批次的页面
                for i in range(batch_start, batch_end):
                    logger.info(f"正在处理物理页面 {i+1}/{total_pages}...")
                    
                    # 保存临时图片
                    temp_path = os.path.join(output_dir, f"temp_page_{i}.jpg")
                    try:
                        # 压缩图片以节省内存
                        image = images[i]
                        image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                        image.save(temp_path, 'JPEG', quality=85)
                        
                        # OCR识别
                        cv_image = cv2.imread(temp_path)
                        ocr_result = self.ocr.ocr(temp_path, rec=True, cls=True)
                        
                        page_info = {
                            "page_number": i + 1,
                            "document_page": None,
                            "tables": [],
                            "content": ""
                        }
                        
                        if ocr_result is not None:
                            structured_ocr = self._process_ocr_results(ocr_result)
                            
                            # 检查是否包含表格
                            if self._is_likely_table(cv_image, structured_ocr):
                                logger.info(f"检测到表格，页面 {i+1}")
                                # 压缩base64图片
                                with Image.open(temp_path) as img:
                                    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                                    buffered = io.BytesIO()
                                    img.save(buffered, format="JPEG", quality=85)
                                    img_data = base64.b64encode(buffered.getvalue()).decode()
                                
                                table_result = self._process_table_with_vision(temp_path, structured_ocr)
                                page_info["tables"].append({
                                    "llm_analysis": table_result,
                                    "base64_image": img_data
                                })
                            
                            # 处理文本内容
                            actual_page_number, content = self._process_page_content(ocr_result[0])
                            if actual_page_number:
                                page_info["document_page"] = actual_page_number
                            page_info["content"] = content
                        
                        batch_result["pages"].append(page_info)
                        
                    except Exception as e:
                        logger.error(f"处理页面 {i+1} 时发生错误: {str(e)}")
                        continue
                    finally:
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                # 保存当前批次结果
                batch_output_path = os.path.join(output_dir, f"result_batch_{batch_start+1}_{batch_end}.json")
                with open(batch_output_path, "w", encoding="utf-8") as f:
                    json.dump(batch_result, f, ensure_ascii=False, indent=2)
                
                # 更新进度
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'last_processed_page': batch_end - 1,
                        'total_pages': total_pages,
                        'last_update': datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"批次 {batch_start + 1} - {batch_end} 处理完成")
            
            # 所有批次处理完成后，可以选择合并所有JSON文件
            self._merge_batch_results(output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"处理PDF时发生错误: {str(e)}")
            raise

    def _merge_batch_results(self, output_dir):
        """合并所有批次结果"""
        try:
            merged_result = {
                "document_info": {
                    "total_pages": 0,
                    "processed_time": datetime.now().isoformat()
                },
                "pages": []
            }
            
            # 查找所有批次结果文件
            batch_files = [f for f in os.listdir(output_dir) 
                         if f.startswith("result_batch_") and f.endswith(".json")]
            
            # 使用自定义排序函数
            def get_batch_number(filename):
                # 从文件名中提取起始页码
                match = re.search(r'result_batch_(\d+)_', filename)
                return int(match.group(1)) if match else 0
            
            # 按照批次号数字大小排序
            batch_files.sort(key=get_batch_number)
            
            logger.info(f"合并顺序: {[get_batch_number(f) for f in batch_files]}")
            
            for batch_file in batch_files:
                file_path = os.path.join(output_dir, batch_file)
                logger.info(f"正在处理批次文件: {batch_file}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    merged_result["pages"].extend(batch_data["pages"])
                    merged_result["document_info"]["total_pages"] = batch_data["document_info"]["total_pages"]
            
            # 保存合并结果
            final_output_path = os.path.join(output_dir, "识别结果_完整.json")
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(merged_result, f, ensure_ascii=False, indent=2)
            
            logger.info("所有批次结果已合并完成")
            
        except Exception as e:
            logger.error(f"合并结果时发生错误: {str(e)}")

    def _process_table_with_vision(self, image_path, structured_ocr):
        """使用视觉模型处理表格"""
        try:
            # 准备OCR文本
            ocr_text = "OCR识别结果如下：\n"
            for item in structured_ocr:
                ocr_text += f"- 位置({item['position'][0]:.1f}, {item['position'][1]:.1f}): {item['text']}\n"
            
            # 准备图像
            pil_image = Image.open(image_path)
            base64_image = self._encode_pil_image(pil_image)
            
            # 构建提示信息
            prompt = f"""这是一个表格图像。我已经使用OCR进行了初步识别，请帮我：
1. 理解表格的整体结构
2. 修正可能的OCR错误
3. 将内容整理成规范的表格格式
4. 特别注意数字区间、小数点等细节的准确性
5. 特别注意合并单元格、跨行、跨列的情况

{ocr_text}

请以Json格式输出完整的、经过整理的表格内容。此外，对于Json未能完整表达的内容和信息，请以文字增加辅助说明。"""
            
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
            logger.error(f"表格处理错误: {str(e)}")
            raise

    def _process_page_content(self, page_result):
        """处理单页内容，提取实际页码并忽略页眉页脚"""
        body = []
        actual_page_number = None
        
        if page_result:
            # 获取页面高度范围
            y_coordinates = []
            for line in page_result:
                box = line[0]
                y_coordinates.extend([box[0][1], box[2][1]])
                
            if y_coordinates:
                min_height = min(y_coordinates)
                max_height = max(y_coordinates)
                page_height = max_height - min_height
                
                # 阈值设置
                header_threshold = min_height + (page_height * 0.07)
                footer_threshold = max_height - (page_height * 0.05)
                
                # 处理每一行文本
                for line in page_result:
                    box = line[0]
                    text = line[1][0].strip()
                    y_middle = (box[0][1] + box[2][1]) / 2
                    
                    # 检查是否是页码
                    if y_middle >= footer_threshold and re.match(r'^\d+$', text):
                        try:
                            actual_page_number = int(text)
                        except ValueError:
                            pass
                    # 忽略页眉区域的内容
                    elif y_middle > header_threshold and y_middle < footer_threshold:
                        body.append(text)
        
        return actual_page_number, "\n".join(body) + "\n"

def main():
    # 使用示例
    pdf_path = "your_pdf_file.pdf"  # 替换为实际的PDF路径
    import os
    # 从环境变量获取API密钥
    dashscope_api_key = os.getenv("DASH_SCOPE_API_KEY","")
    parser = EnhancedPDFParser(dashscope_api_key)
    try:
        result_text = parser.process_pdf_with_tables(pdf_path)
        print("处理完成！")
        print(f"识别结果已保存在: output/识别结果.json")
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()
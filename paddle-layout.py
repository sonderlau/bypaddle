from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import os
import re

def process_pdf_with_layout(pdf_path):
    """
    使用PaddleOCR处理PDF文件，关注实际页码
    """
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False,
        show_log=False,
        rec=True,
        det=True
    )
    
    try:
        print("正在转换PDF为图片...")
        images = convert_from_path(
            pdf_path,
            dpi=300,
            fmt='jpg'
        )
        
        full_text = ""
        
        for i, image in enumerate(images):
            print(f"正在处理物理页面 {i+1}/{len(images)}...")
            
            temp_path = f"temp_page_{i}.jpg"
            image.save(temp_path, 'JPEG', quality=95)
            
            result = ocr_engine.ocr(temp_path, rec=True, cls=True)
            
            if result is not None:
                actual_page_number, content = process_page_content(result[0])
                page_header = f"\n=== 物理页 {i+1}"
                if actual_page_number:
                    page_header += f" (文档页 {actual_page_number})"
                page_header += " ===\n"
                full_text += page_header + content
            
            os.remove(temp_path)
            
        return full_text
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return ""
def process_page_content(page_result):
    """
    处理单页内容，提取实际页码并忽略页眉
    返回: (实际页码, 页面内容)
    """
    body = []
    actual_page_number = None
    
    if page_result:
        # 获取页面高度范围
        y_coordinates = []
        for line in page_result:
            box = line[0]  # box是一个包含四个点坐标的列表
            # 添加上下两个y坐标
            y_coordinates.extend([box[0][1], box[2][1]])
        
        if y_coordinates:
            min_height = min(y_coordinates)
            max_height = max(y_coordinates)
            page_height = max_height - min_height
            
            # 阈值设置
            header_threshold = min_height + (page_height * 0.07)
            footer_threshold = max_height - (page_height * 0.05)
            
            # 在process_page_content函数中添加这些打印语句
            print(f"Debug - 页面高度范围: {min_height} to {max_height}")
            print(f"Debug - 页眉阈值: {header_threshold}")
            print(f"Debug - 页脚阈值: {footer_threshold}")
            # 处理每一行文本
            for line in page_result:
                box = line[0]
                text = line[1][0].strip()
                y_middle = (box[0][1] + box[2][1]) / 2  # 使用第一个点和第三个点的y坐标平均值
                
                # 检查是否是页码（在页脚位置的数字）
                if y_middle >= footer_threshold and re.match(r'^\d+$', text):
                    try:
                        actual_page_number = int(text)
                    except ValueError:
                        pass
                # 忽略页眉区域的内容
                elif y_middle > header_threshold and y_middle < footer_threshold:
                    body.append(text)
    
    return actual_page_number, "\n".join(body) + "\n"

def create_page_mapping(text, mapping_path):
    """
    创建物理页码和实际页码的映射文件
    """
    mapping = []
    for line in text.split('\n'):
        if line.startswith('=== 物理页'):
            mapping.append(line)
    
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mapping))
    print(f"页码映射已保存到 {mapping_path}")

def main():
    pdf_path = "你的PDF文件路径.pdf"  # 替换为实际的PDF路径
    output_path = "识别结果.txt"
    
    try:
        print("开始处理PDF...")
        text = process_pdf_with_layout(pdf_path)
        
        # 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"转换完成！结果已保存到 {output_path}")
        
        # 创建页码映射文件
        mapping_path = "页码映射.txt"
        create_page_mapping(text, mapping_path)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
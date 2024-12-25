from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import os
import json

def process_pdf_with_layout(pdf_path, save_visualization=False):
    """
    使用PaddleOCR Layout处理PDF文件
    """
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False,
        show_log=False,
        rec=True,
        det=True,
        table=True,
        layout=True
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
            print(f"正在处理第 {i+1}/{len(images)} 页...")
            
            temp_path = f"temp_page_{i}.jpg"
            image.save(temp_path, 'JPEG', quality=95)
            
            # 进行OCR识别
            result = ocr_engine.ocr(temp_path, rec=True, cls=True)
            
            # 提取文本
            if result is not None:
                page_text = extract_text_from_result(result)
                full_text += f"\n=== 第 {i+1} 页 ===\n{page_text}\n"
            
            os.remove(temp_path)
            
        return full_text
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise

# 在 extract_text_from_result 函数中添加调试信息
def extract_text_from_result(result):
    """
    从OCR结果中提取文本，带调试信息
    """
    text = ""
    try:
        print("调试信息 - OCR结果类型:", type(result))
        print("调试信息 - OCR结果前100个字符:", str(result)[:100])
        
        if isinstance(result, list) and len(result) > 0:
            for item in result:
                print("调试信息 - 文本块类型:", type(item))
                if isinstance(item, list):
                    for line in item:
                        print("调试信息 - 行内容类型:", type(line))
                        if isinstance(line, dict) and 'text' in line:
                            text += line['text'] + "\n"
                        elif isinstance(line, list) and len(line) > 1:
                            text += line[1][0] + "\n"
    except Exception as e:
        print(f"提取文本时出错: {str(e)}")
    return text

def main():
    pdf_path = "111.pdf"  # 替换为实际的PDF路径
    output_path = "识别结果.txt"
    
    try:
        # 处理PDF
        print("开始处理PDF...")
        text = process_pdf_with_layout(pdf_path)
        
        # 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"转换完成！结果已保存到 {output_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
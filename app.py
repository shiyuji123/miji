from flask import Flask, request, jsonify, render_template
from PIL import Image
import pytesseract
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from embedding_utils import get_most_similar_doc
import base64
import tempfile

app = Flask(__name__, template_folder='templates')  # 指定模板目录

@app.route("/")
def home():
    return render_template("index.html")  # 载入 index.html 前端页面

# 加载环境变量
load_dotenv('.env')

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.point(lambda x: 0 if x < 128 else 255)
        text = pytesseract.image_to_string(img, lang='eng+chi_sim', config='--psm 6 --oem 3')
        return text.strip()
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None

@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # 直接读取图像对象（避免临时文件冲突）
        img = Image.open(file.stream)
        img = img.convert('L')
        img = img.point(lambda x: 0 if x < 128 else 255)
        text = pytesseract.image_to_string(img, lang='eng+chi_sim', config='--psm 6 --oem 3')
        return jsonify({"text": text.strip()})
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500


@app.route("/ask_question", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # 获取嵌入向量
    try:
        resp = client.embeddings.create(
            model="doubao-embedding-large-text-240915",
            input=[question],
            encoding_format="float"
        )
        question_vec = resp.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"Embedding error: {str(e)}"}), 500

    # 获取最相关文档
    try:
        best_file, similarity = get_most_similar_doc('vector_store', question_vec)
        with open(f'documents/{os.path.splitext(best_file)[0]}.txt', 'r', encoding='utf-8') as f:
            related_content = f.read()
    except Exception as e:
        return jsonify({"error": f"Document retrieval error: {str(e)}"}), 500

    # 构建系统提示词并生成回答
    try:
        system_prompt = f"""你是一个专业助手，请基于以下文档内容回答问题:
\n{related_content}\n
回答需专业、准确，如不清楚请说明。"""

        completion = client.chat.completions.create(
            model="deepseek-v3-241226",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"Completion error: {str(e)}"}), 500

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)

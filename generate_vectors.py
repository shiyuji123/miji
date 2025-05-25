# generate_vectors.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env')

client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

if not os.path.exists('vector_store'):
    os.makedirs('vector_store')

for file in os.listdir('documents'):
    if not file.endswith('.txt'):
        continue

    with open(f'documents/{file}', 'r', encoding='utf-8') as f:
        content = f.read()

    resp = client.embeddings.create(
        model="doubao-embedding-large-text-240915",
        input=[content],
        encoding_format="float"
    )

    embedding = resp.data[0].embedding
    with open(f'vector_store/{os.path.splitext(file)[0]}.json', 'w') as out_f:
        json.dump(embedding, out_f)

    print(f"嵌入完成: {file}")

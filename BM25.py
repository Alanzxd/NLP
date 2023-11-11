import pandas as pd
from rank_bm25 import BM25Okapi
import json

# 读取TSV文件
df = pd.read_csv('popQA.tsv', sep='\t')

# 提取问题和构建语料库
questions = df['question'].tolist()
question_ids = df['id'].tolist()
possible_answers = df['possible_answers'].apply(lambda x: eval(x))  # 将字符串转换为列表

# 构建语料库（这里我们使用subj, prop, obj列拼接成文本）
corpus = (df['subj'] + ' ' + df['prop'] + ' ' + df['obj']).tolist()

# 使用BM25处理语料库
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 创建JSONL输出
with open('popQA_BM25_OUTPUT.jsonl', 'w') as f:
    for idx, question in enumerate(questions):
        tokenized_query = question.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_docs = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:5]
        
        ctxs = []
        for doc_id in top_docs:
            text = corpus[doc_id]
            has_answer = any(answer.lower() in text.lower() for answer in possible_answers[idx])  # 检查答案是否出现在文本中

            ctxs.append({
                "id": df.iloc[doc_id]['id'],  # 从数据集获取文档的ID
                "text": text,
                "hasanswer": has_answer
            })

        retrieval_output = {
            "id": question_ids[idx],
            "question": question,
            "ctxs": ctxs
        }
        json.dump(retrieval_output, f)
        f.write('\n')

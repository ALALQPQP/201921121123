# 201921121123
# 论文查重算法实现

我们将使用Python3进行项目实现。首先，我们需要设计一个论文查重算法，我们选择使用余弦相似度进行计算。

## 余弦相似度算法

余弦相似度基于词袋模型，其公式如下:
\[
\mathrm{cosine\ similarity}=\frac{A\cdot B}{\parallel A\parallel\parallel B\parallel}
\]
其中:
- \(A\) 和 \(B\) 是两个文档的向量表示。
- \(⋅\) 表示向量的点积。
- \(\parallel A\parallel\) 和 \(\parallel B\parallel\) 是向量的模。

## 代码实现

### 文本预处理函数

```python
import re
from typing import List

def preprocess_text(text: str) -> List[str]:
    """
    预处理文本: 清洗和标准化。
    """
    # 移除所有非中文字符
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    
    # 将文本分为单独的字符
    words = list(text)
    
    return words
### 计算余弦相似度函数
```python
from collections import Counter
import numpy as np

def calculate_cosine_similarity(doc1_words: List[str], doc2_words: List[str]) -> float:
    """
    计算两篇文档的余弦相似度。
    """
    # 为两篇文档创建词频字典
    doc1_word_freq = Counter(doc1_words)
    doc2_word_freq = Counter(doc2_words)
    
    # 获取所有词的集合
    all_words = set(doc1_word_freq.keys()).union(set(doc2_word_freq.keys()))
    
    # 创建词向量
    doc1_vector = np.array([doc1_word_freq.get(word, 0) for word in all_words])
    doc2_vector = np.array([doc2_word_freq.get(word, 0) for word in all_words])
    
    # 计算余弦相似度
    cosine_similarity = np.dot(doc1_vector, doc2_vector) / (np.linalg.norm(doc1_vector) * np.linalg.norm(doc2_vector))
    
    return cosine_similarity
### 主函数
```python
import argparse

def main():
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description="计算两篇文档的相似度。")
    
    parser.add_argument("orig_file_path", help="原文文件的绝对路径。")
    parser.add_argument("copy_file_path", help="抄袭版论文的文件的绝对路径。")
    parser.add_argument("output_file_path", help="输出答案文件的绝对路径。")
    
    args = parser.parse_args()
    
    # 读取原文和抄袭版的内容
    with open(args.orig_file_path, 'r', encoding='utf-8') as orig_file, open(args.copy_file_path, 'r', encoding='utf-8') as copy_file:
        orig_text = orig_file.read()
        copy_text = copy_file.read()
    
    orig_words = preprocess_text(orig_text)
    copy_words = preprocess_text(copy_text)
    
    similarity_rate = calculate_cosine_similarity(orig_words, copy_words)
    
    # 将相似度写入输出文件
    with open(args.output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{similarity_rate:.2f}")

if __name__ == "__main__":
    main()
### 使用
要使用此程序，可以在命令行中运行以下命令:
python main.py "path/to/original_text.txt" "path/to/copied_text.txt" "path/to/output_similarity.txt"


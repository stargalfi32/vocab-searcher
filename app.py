from flask import Flask, render_template, request
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

app = Flask(__name__)

# 下載 NLTK 所需的斷詞與詞性還原資料包
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 全域快取，避免重複進行磁碟讀取與 NLTK 斷詞運算
PROCESSED_PASSAGES = None

def load_passages(folder_path):
    passages = []
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found.")
        return None

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 確保只處理檔案，忽略子資料夾（如隱藏的 .git 或其他資料夾）
        if os.path.isdir(file_path):
            continue
            
        category = os.path.splitext(file_name)[0]
        
        # 嘗試使用多種常見編碼讀取檔案，防止 UnicodeDecodeError
        content = None
        for enc in ['utf-8', 'big5', 'utf-16', 'gbk', 'utf-8-sig']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
                
        if content is None:
            print(f"無法讀取檔案 {file_name}，編碼不支援。")
            continue

        for line in content.splitlines():
            passage = line.strip()
            if passage:  # 確保不是空行
                passages.append({'category': category, 'passage': passage})
    return passages

def clean_mcq(mcq):
    return re.sub(r'^\d+\.\s*', '', mcq)

def highlight_word(text, word):
    stemmer = PorterStemmer()
    word_stem = stemmer.stem(word.lower())

    def replacer(match):
        token = match.group(0)
        if stemmer.stem(token.lower()) == word_stem:
            return f'<span class="highlight">{token}</span>'
        return token

    return re.sub(r'\w+', replacer, text)

def sort_matches(matches):
    def sort_key(item):
        cat = item['category']
        num_match = re.search(r'\d+', cat)
        num = int(num_match.group()) if num_match else 0
        exam_type = 0 if '學測' in cat else 1
        return (exam_type, -num)
    return sorted(matches, key=sort_key)

def sort_categories(categories):
    def sort_key(cat):
        num_match = re.search(r'\d+', cat)
        num = int(num_match.group()) if num_match else 0
        exam_type = 0 if '學測' in cat else 1
        return (exam_type, -num)
    return sorted(categories, key=sort_key)

def pre_process_passages(folder_path):
    raw_passages = load_passages(folder_path)
    if raw_passages is None:
        return []
        
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    mcq_pattern = re.compile(r'(\d+\.\s.*?)(\(A\).*?\(B\).*?\(C\).*?\(D\).*?)(?=\n|$)')
    
    processed = []
    for entry in raw_passages:
        category = entry['category']
        text = entry['passage']
        
        # 1. 處理選擇題 (MCQ)
        mcqs = mcq_pattern.findall(text)
        for mcq in mcqs:
            question = mcq[0]
            options = mcq[1]
            combined_mcq = question + options
            words = word_tokenize(combined_mcq)
            
            processed.append({
                'cleaned_text': clean_mcq(combined_mcq.strip()),
                'category': category,
                'words_lower': [w.lower() for w in words],
                'stems': [stemmer.stem(w).lower() for w in words],
                'lemmas': [lemmatizer.lemmatize(w).lower() for w in words]
            })
            
        # 2. 移除選擇題後的其他文本，用句子斷詞
        text_without_mcq = re.sub(mcq_pattern, '', text)
        sentences = sent_tokenize(text_without_mcq)
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            processed.append({
                'cleaned_text': sentence.strip(),
                'category': category,
                'words_lower': [w.lower() for w in words],
                'stems': [stemmer.stem(w).lower() for w in words],
                'lemmas': [lemmatizer.lemmatize(w).lower() for w in words]
            })
            
    return processed

def get_processed_passages():
    global PROCESSED_PASSAGES
    if PROCESSED_PASSAGES is None:
        PROCESSED_PASSAGES = pre_process_passages('passages')
    return PROCESSED_PASSAGES

def search_sentences_optimized(word, selected_types):
    exact_matches = []
    related_matches = []
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    word_lower = word.lower()
    word_stem = stemmer.stem(word_lower)
    word_lemma = lemmatizer.lemmatize(word_lower)
    
    passages = get_processed_passages()
    
    for item in passages:
        category = item['category']
        # 若使用者只選擇學測或指考，篩選分類
        if not any(t in category for t in selected_types):
            continue
            
        # 精確匹配
        if word_lower in item['words_lower']:
            highlighted = highlight_word(item['cleaned_text'], word)
            exact_matches.append({'sentence': highlighted, 'category': category})
        # 相關匹配 (字根或原形相同)
        elif word_stem in item['stems'] or word_lemma in item['lemmas']:
            highlighted = highlight_word(item['cleaned_text'], word)
            related_matches.append({'sentence': highlighted, 'category': category})
            
    return sort_matches(exact_matches), sort_matches(related_matches)

def display_results(exact_matches, related_matches):
    exact_categories = {m['category'] for m in exact_matches}
    related_categories = {m['category'] for m in related_matches}
    return sort_categories(exact_categories), sort_categories(related_categories)

@app.route('/', methods=['GET', 'POST'])
def index():
    # 每次請求時，確保至少初始化過快取
    get_processed_passages()

    if request.method == 'POST':
        word = request.form['word']
        selected_types = request.form.getlist('exam_type')  # e.g., ["學測", "指考"]

        if word.lower() == 'exit':
            return render_template('index.html')

        if not selected_types:
            return render_template('index.html', word=word, error="請至少選擇一個考試類型")

        exact_matches, related_matches = search_sentences_optimized(word, selected_types)
        exact_categories, related_categories = display_results(exact_matches, related_matches)

        return render_template('index.html',
                               word=word,
                               exact_matches=exact_matches,
                               related_matches=related_matches,
                               exact_categories=exact_categories,
                               related_categories=related_categories,
                               selected_types=selected_types)

    return render_template('index.html', selected_types=["學測", "指考"])  # 預設兩個都選

if __name__ == '__main__':
    app.run(debug=True)

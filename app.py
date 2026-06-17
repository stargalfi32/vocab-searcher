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

def search_sentences(passages, word, selected_types):
    exact_matches = []
    related_matches = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    mcq_pattern = re.compile(r'(\d+\.\s.*?)(\(A\).*?\(B\).*?\(C\).*?\(D\).*?)(?=\n|$)')

    for entry in passages:
        category = entry['category']
        # 若使用者只選擇學測或指考，篩選分類
        if not any(t in category for t in selected_types):
            continue

        text = entry['passage']
        mcqs = mcq_pattern.findall(text)

        for mcq in mcqs:
            question = mcq[0]
            options = mcq[1]
            combined_mcq = question + options
            words = word_tokenize(combined_mcq)

            stems = [stemmer.stem(w).lower() for w in words]
            lemmas = [lemmatizer.lemmatize(w).lower() for w in words]

            highlighted = highlight_word(clean_mcq(combined_mcq.strip()), word)
            if word.lower() in [w.lower() for w in words]:
                exact_matches.append({'sentence': highlighted, 'category': category})
            elif stemmer.stem(word.lower()) in stems or lemmatizer.lemmatize(word.lower()) in lemmas:
                related_matches.append({'sentence': highlighted, 'category': category})

        text = re.sub(mcq_pattern, '', text)
        sentences = sent_tokenize(text)

        for sentence in sentences:
            words = word_tokenize(sentence)
            stems = [stemmer.stem(w).lower() for w in words]
            lemmas = [lemmatizer.lemmatize(w).lower() for w in words]

            highlighted = highlight_word(sentence.strip(), word)
            if word.lower() in [w.lower() for w in words]:
                exact_matches.append({'sentence': highlighted, 'category': category})
            elif stemmer.stem(word.lower()) in stems or lemmatizer.lemmatize(word.lower()) in lemmas:
                related_matches.append({'sentence': highlighted, 'category': category})

    return sort_matches(exact_matches), sort_matches(related_matches)

def display_results(exact_matches, related_matches):
    exact_categories = {m['category'] for m in exact_matches}
    related_categories = {m['category'] for m in related_matches}
    return sort_categories(exact_categories), sort_categories(related_categories)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        word = request.form['word']
        selected_types = request.form.getlist('exam_type')  # e.g., ["學測", "指考"]

        if word.lower() == 'exit':
            return render_template('index.html')

        if not selected_types:
            return render_template('index.html', word=word, error="請至少選擇一個考試類型")

        passages = load_passages('passages')
        if passages is None:
            return "Passages folder not found."

        exact_matches, related_matches = search_sentences(passages, word, selected_types)
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

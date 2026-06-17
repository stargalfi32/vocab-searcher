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
nltk.download('stopwords')
from nltk.corpus import stopwords

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

        # 直接讀取整份檔案的內容，不以行為單位拆分，以便跨行比對題目與選項
        passages.append({'category': category, 'passage': content})
    return passages

def clean_mcq(mcq):
    # 移除題號 (例如 "1. " 或 "12 ")
    return re.sub(r'^\d+[\.\s]+', '', mcq)

def highlight_word(text, word):
    stemmer = PorterStemmer()
    word_stem = stemmer.stem(word.lower())

    def replacer(match):
        token = match.group(0)
        if stemmer.stem(token.lower()) == word_stem:
            return f'<span class="highlight">{token}</span>'
        return token

    # 只針對單字字元進行替換，防範 HTML 標籤損毀與基本 XSS 注入
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
    
    # 嚴謹的選擇題正則表達式，支援跨行且防止跨題號過度匹配，限制題號必須位於行首或文章開頭
    mcq_pattern = re.compile(
        r'(?:\A|(?<=\n))(\s*\d+[\.\s]+(?:(?!\n\s*\d+[\.\s]|\(?A\)|（A）|A\.).)*?)'  # 題號與題幹 (Group 1)
        r'(\s*(?:\(A\)|（A）|A\.)(?:(?!\n\s*\d+[\.\s]|\(?B\)|（B）|B\.).)*?' # 選項 A (Group 2)
        r'\s*(?:\(B\)|（B）|B\.)(?:(?!\n\s*\d+[\.\s]|\(?C\)|（C）|C\.).)*?' # 選項 B
        r'\s*(?:\(C\)|（C）|C\.)(?:(?!\n\s*\d+[\.\s]|\(?D\)|（D）|D\.).)*?' # 選項 C
        r'\s*(?:\(D\)|（D）|D\.)[^\n]*)',                               # 選項 D
        re.DOTALL | re.IGNORECASE
    )
    
    # 英文停用詞，用於配對選項單字時過濾常見虛詞，防止誤判
    stop_words = set(stopwords.words('english'))
    
    processed = []
    for entry in raw_passages:
        category = entry['category']
        text = entry['passage'].replace('\r\n', '\n')
        
        # 先將文章切出一般段落的句子，便於之後將選擇題選項與題目句進行匹配還原
        mcqs_raw = mcq_pattern.findall(text)
        text_without_mcq = re.sub(mcq_pattern, '', text)
        paragraphs = re.split(r'\n\s*\n', text_without_mcq)
        sentences = []
        for para in paragraphs:
            if para.strip():
                sentences.extend([s.strip() for s in sent_tokenize(para) if s.strip()])
                
        # 1. 提取並處理選擇題 (MCQ)
        for mcq in mcqs_raw:
            question = mcq[0]
            options = mcq[1]
            
            q_num_match = re.search(r'\d+', question)
            if q_num_match:
                q_num = q_num_match.group()
                
                # 解析出選項單字
                opts = re.split(r'\s*(?:\([A-J]\)|（[A-J]）|[A-J]\.)\s*', options.strip())
                opt_words = [o.strip() for o in opts if o.strip()]
                
                # 尋找原始題目句
                matched_sentence = None
                reconstructed_sentence = None
                
                # A. 優先使用題號空格比對，例如： "  30  ", " [30] ", " (30) "
                blank_pattern = re.compile(
                    rf'(?:[＿_\[\(\u3010\u3014\u2014]|\s{{2,}}){q_num}(?:[＿_\]\)\u3011\u3015\u2014]|\s{{2,}}|[\.,;\?!\s]{{2,}})'
                )
                for s in sentences:
                    if blank_pattern.search(s):
                        matched_sentence = s
                        reconstructed_sentence = s
                        break
                        
                # B. 備用方案：如果文章已經被填入了正確答案（無空格），比對選項單字（排除停用詞）
                if not matched_sentence:
                    for s in sentences:
                        s_words = [w.lower() for w in word_tokenize(s)]
                        s_stems = [stemmer.stem(w).lower() for w in s_words]
                        s_lemmas = [lemmatizer.lemmatize(w).lower() for w in s_words]
                        
                        for opt in opt_words:
                            opt_lower = opt.lower()
                            if opt_lower in stop_words:
                                continue
                            
                            opt_words_list = word_tokenize(opt_lower)
                            if len(opt_words_list) > 1:
                                # 片語匹配
                                if opt_lower in s.lower():
                                    matched_sentence = s
                                    reconstructed_sentence = re.sub(rf'\b{re.escape(opt)}\b', f"   {q_num}   ", s, flags=re.IGNORECASE)
                                    break
                            else:
                                # 單字匹配
                                opt_stem = stemmer.stem(opt_lower)
                                opt_lemma = lemmatizer.lemmatize(opt_lower)
                                if opt_lower in s_words or opt_stem in s_stems or opt_lemma in s_lemmas:
                                    matched_word = None
                                    for w in word_tokenize(s):
                                        w_lower = w.lower()
                                        if w_lower == opt_lower or stemmer.stem(w_lower) == opt_stem or lemmatizer.lemmatize(w_lower) == opt_lemma:
                                            matched_word = w
                                            break
                                    if matched_word:
                                        matched_sentence = s
                                        reconstructed_sentence = re.sub(rf'\b{re.escape(matched_word)}\b', f"   {q_num}   ", s)
                                        break
                        if matched_sentence:
                            break
                            
                # 若成功配對出題目句，將其與選項合併；否則只合併原本的題幹與選項
                if reconstructed_sentence:
                    combined_mcq = reconstructed_sentence.strip() + "\n" + options.strip()
                else:
                    combined_mcq = question.strip() + "\n" + options.strip()
            else:
                combined_mcq = question.strip() + "\n" + options.strip()
                
            words = word_tokenize(combined_mcq)
            processed.append({
                'cleaned_text': clean_mcq(combined_mcq),
                'category': category,
                'words_lower': [w.lower() for w in words],
                'stems': [stemmer.stem(w).lower() for w in words],
                'lemmas': [lemmatizer.lemmatize(w).lower() for w in words]
            })
            
        # 2. 將文章非選擇題句本身也獨立加進來
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
    # 確保已載入快取
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

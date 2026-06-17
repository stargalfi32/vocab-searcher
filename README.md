# 學測單字搜尋器 (Vocab Searcher)

這是一個基於 Flask 與 NLTK 的英文單字句子搜尋系統，支援學測與指考題目的句子檢索與詞形還原（Lemmatization/Stemming）匹配。

## 目錄結構
```
vocab-searcher/
├── app.py                   # Flask 後端主程式
├── requirements.txt         # 專案套件依賴檔
├── templates/
│   └── index.html           # 前端 HTML 範本
└── passages/
    └── 111學測_english.txt   # 文章/考題資料檔 (可自行新增其他 .txt 檔)
```

## 本地執行步驟

1. 安裝必要套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 執行 Flask 伺服器：
   ```bash
   python app.py
   ```

3. 在瀏覽器打開連結：
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 線上部署說明

本專案已配置好 `requirements.txt`，支援部署至 **PythonAnywhere** 或 **Render** 等雲端平台。詳細步驟請參考 AI 的回答說明。

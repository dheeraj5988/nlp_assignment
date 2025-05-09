# Real-time QoS Optimization in Game Streaming using Sentiment Analysis

**Name:** Dheeraj Sharma  
**Reg No:** RA2211027010017  
**Section:** AD1  

## 🔍 Problem Statement
Live video game streaming platforms like Twitch face fluctuating QoS due to changing network conditions and user engagement. This project proposes a deep learning + NLP pipeline to analyze live chat data, classify sentiment using BERT, and predict satisfaction using an LSTM-based DNN.

## 💾 Dataset
We used Twitch chat messages (sample: `healthygamer_gg_testdata.csv`). Each message is classified with:
- **Sentiment (positive/negative)** via HuggingFace BERT.
- **Satisfaction** derived from sentiment (binary class for DNN).

## 🛠️ Technologies Used
- Python, Pandas, Matplotlib
- HuggingFace Transformers (BERT)
- TensorFlow/Keras
- Scikit-learn

## 📁 Project Structure
```
├── .gitignore
├── analyze_chat.py
├── bert_sentimitize.py
├── eval_bert_model.py
├── eval_my_model.py
├── labeled_dataset.csv
├── my_model.keras
├── mymodel_sentimitize.py
├── readme.md
```

## ⚙️ How to Run

1. **Label Sentiment:**
   ```bash
   python bert_sentimitize.py
   ```

2. **Train DNN Model:**
   ```bash
   python mymodel_sentimitize.py
   ```

3. **Analyze Chat (WordCloud):**
   ```bash
   python analyze_chat.py
   ```

4. **Evaluate Models:**
   ```bash
   python eval_bert_model.py
   python eval_my_model.py
   ```

## 🎯 Outcome
A dual-model pipeline where BERT classifies sentiment and a DNN classifies satisfaction. Can be extended for real-time stream feedback integration for QoS management.

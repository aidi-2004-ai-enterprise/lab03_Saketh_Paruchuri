# lab03_Saketh_Paruchuri
# 🐧 Lab 3: Penguins Classification with XGBoost and FastAPI

---

## 🎞️ Demo 

![Demo](recording/demo.gif)

> 📹 A  screen recording (`demo.mp4`) is also available inside the `recording/` folder.

---

This project is part of **AIDI 2004 – AI Enterprise Applications**.  
It demonstrates a complete end-to-end machine learning pipeline using the **Seaborn Penguins dataset**. The application trains an XGBoost model to predict penguin species and serves predictions via a RESTful API using FastAPI.

---

## 🚀 How to Run

1. **Install dependencies** using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Train the model**:
   ```bash
   python train.py
   ```

3. **Run the API server**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access API documentation**:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   
---

## 📁 Project Structure

```
lab03_Saketh_Paruchuri/
├── app/
│   ├── main.py               # FastAPI application with /predict and /health endpoints
│   └── data/
│       ├── model.json        # Trained XGBoost model
│       └── target_classes.csv  # List of encoded class labels
├── train.py                  # Script to train and export the model
├── requirements.txt          # Required Python packages
├── pyproject.toml            # Optional uv/poetry config
├── recording/
│   ├── demo.gif              # GIF of working application
│   └── demo.mp4              # Full screen recording (compressed)
└── README.md                 # Project documentation (you are here)
```



## ✅ Features

- 🔍 Uses Seaborn’s penguin dataset for multi-class classification.
- 🧠 Trained with XGBoost Classifier with label and one-hot encoding.
- 🚀 FastAPI backend with real-time prediction API (`/predict`).
- ✅ Input validation using Pydantic.
- 🔁 Graceful error handling (422).
- 🎥 Demo GIF and screen recording included.

---

## 👤 Author

**Saketh Paruchuri**  
Postgraduate Student – Durham College  
Course: AIDI 2004 – AI Enterprise Applications  
🔗 GitHub Repo: [lab03_Saketh_Paruchuri](https://github.com/aidi-2004-ai-enterprise/lab03_Saketh_Paruchuri)

---

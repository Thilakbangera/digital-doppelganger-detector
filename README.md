# digital-doppelganger-detector

# 🤖 Digital Doppelgänger Detector

A powerful AI-based application to detect **deepfakes** across **text, images, and videos** — all in one place. Built with Python, Streamlit, and multiple state-of-the-art models to ensure media authenticity.

---

## 🚀 Features

- 🧠 **Text Deepfake Detection** — Detects AI-generated text using fine-tuned transformers.
- 🖼️ **Image Deepfake Detection** — Uses CNN/ResNet models to classify real vs fake images.
- 🎥 **Video Deepfake Detection** — Leverages EfficientNet (DFDC challenge) models for detecting manipulated videos.
- 🧪 Clean, modern **Streamlit-based UI**
- 🔍 Real-time predictions with user-friendly file upload system
- 📊 Output probability scores + verdict (REAL / FAKE)

---

## 📂 Folder Structure

```

digital-doppelganger-detector/
├── app.py                            # Main Streamlit UI
├── requirements.txt                 # Python dependencies
├── README.md                        # You're here!
├── models/
│   ├── text\_model/                  # Text classification model
│   ├── image\_model/                 # ResNet or custom CNN for images
│   └── dfdc\_deepfake\_challenge/    # EfficientNet-based video model
│       ├── weights/                # Model weights (\*.pth)
│       ├── libs/                   # Helper files (e.g., landmark predictor)
│       └── predict\_folder.py       # Main script for batch video prediction
├── utils/
│   └── predict.py                   # Helper for handling multi-modal predictions
├── output/                          # Contains the demo video (demo.mp4)

````

---

## 🧠 Models Used

### 1. **Text Model**
- Type: BERT / Transformer classifier
- Task: Classify human vs AI-generated content
- Input: Text
- Output: Real / Fake + confidence

### 2. **Image Model**
- Type: ResNet18 / Custom CNN
- Dataset: DeepFake detection image dataset
- Input: Image files
- Output: Real / Fake classification

### 3. **Video Model**
- Type: EfficientNet-B7 (from DFDC Challenge)
- Dataset: Kaggle Deepfake Detection Challenge
- Input: Video file (.mp4)
- Output: CSV with predictions + real/fake flag

---

## 💻 Installation & Setup

> ⚠️ Python 3.9 is required.

### 1. Clone the repository:

```bash
git clone https://github.com/Thilakbangera/digital-doppelganger-detector.git
cd digital-doppelganger-detector
````

### 2. Create virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # (Windows)
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

This launches a browser window with a UI to test:

* **Text**: Paste any content
* **Image**: Upload `.jpg`, `.png`
* **Video**: Upload `.mp4` (model will take a few seconds to process)

---

## 📈 Video Deepfake Detection (DFDC)

For video predictions:

* All predictions use `models/dfdc_deepfake_challenge/predict_folder.py`
* Place `.mp4` videos inside `test_videos/`
* Outputs go to `submission.csv` with `video_name, label`

---

## 📦 Git LFS Notice

Some model weights are large (>100MB). This repo uses [Git LFS](https://git-lfs.github.com/) to manage large files.

```bash
git lfs install
git lfs pull
```

---

## 🎬 Demo Video

Curious how it works? Check out the full walkthrough video:

📁 The demo video is included in the `output/` directory:  
**`output/demo.mp4`**

> 🎥 This video showcases real-time detection across text, image, and video formats.

---

## 📚 Citation / References

* [DeepFake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge)
* [FaceForensics++](https://github.com/ondyari/FaceForensics)
* [ResNet - He et al.](https://arxiv.org/abs/1512.03385)
* [Streamlit Docs](https://docs.streamlit.io)

---

## 🤝 Contributing

Pull requests, suggestions, and issues are welcome! Please open an issue to discuss any major changes.


#  moji - handwritten digit recognition

**[moji live link](https://moji.up.railway.app/)**

A simple web application that recognizes handwritten digits (0-9) using machine learning. Draw a digit on your screen, and the AI predicts what you wrote!

---

## What You Need

- Python 3.7 or higher
- pip (Python package manager)
- A web browser

---

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)
```bash
python train_model.py
```
This trains the neural network on the MNIST dataset and saves it as `digit_model.h5`. Takes about 5-10 minutes. You only need to do this once!

### Step 3: Start the Application
```bash
python backend.py
```
Then open `index.html` in your web browser.

---

## How to Use

1. **Draw** a digit (0-9) on the canvas using your mouse
2. **Click "Predict"** to see what the AI thinks you drew
3. **Click "Clear"** to erase and try again


---

## Technical Details

- **Backend**: Python Flask server running on `http://localhost:8080`
- **Frontend**: HTML/CSS/JavaScript for drawing interface
- **Model**: Convolutional Neural Network (CNN) trained on MNIST handwritten digit dataset
- **Framework**: TensorFlow/Keras

---

## 📁 Project Structure

```
moji/
├── backend.py          # Flask server for predictions
├── train_model.py      # Script to train the neural network
├── index.html          # Frontend (draw interface)
├── digit_model.h5      # Trained AI model (created after training)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## Troubleshooting

**Model not found error?**
- Run `python train_model.py` first

**Can't connect to backend?**
- Make sure `python backend.py` is running
- Check that port 8080 is available

**Drawing doesn't work?**
- Try a different browser (Chrome, Firefox, Safari)
- Make sure JavaScript is enabled

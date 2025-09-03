# MNIST Interactive Digit Recognizer (with CNN)

This project implements a simple **Convolutional Neural Network (CNN)** trained on the MNIST dataset to recognize handwritten digits.  
It includes training the model, saving it, and testing it interactively through a drawing interface.

---

## Project Structure

- **model.py**  
  Contains the `CNNModel` class (Convolutional Neural Network).  
  This is the architecture used for MNIST digit recognition.

- **train.py**  
  Trains the CNN on the MNIST dataset. Steps:  
  1. Load the MNIST dataset (training + test).  
  2. Define the CNN (`CNNModel`).  
  3. Train the model using Adam optimizer and CrossEntropyLoss.  
  4. Track accuracy and loss during training.  
  5. Save the trained weights to `mnist_cnn.pth`.  
  6. Generate a plot showing accuracy progression from 1 to 50 epochs.

- **draw.py**  
  Interactive digit testing with Tkinter:  
  - Draw a digit on a 280×280 canvas.  
  - Click **Predict** to see the model’s guess.  
  - Click **Clear** to reset the canvas.  

⚠️ Note: While the CNN is more robust than a simple dense network, it still expects digits to be centered and relatively similar to MNIST’s style.

---

## Requirements

- Python 3.10+  
- Required libraries:

```bash
  pip install torch torchvision matplotlib pillow
````

---

## Usage

### 1. Train the CNN

```bash
python train.py
```

- The model will train on MNIST and save the weights to `mnist_cnn.pth`.
- A plot of training accuracy (1–50 epochs) will be generated.

### 2. Test the model interactively

```bash
python draw.py
```

- A Tkinter window will open with a black canvas.
- Draw a digit using your mouse (white strokes).
- Click **Predict** to see the recognized digit.
- Click **Clear** to draw again.

---

## Notes

- MNIST uses **black background and white digits**. The Tkinter interface follows the same convention.
- If the drawn digit is too small, off-center, or unusual, the prediction may fail.
- Preprocessing (centering, scaling) and data augmentation during training can further improve results.

---

## Future Improvements

- Add a real-time 28×28 preview of what the CNN “sees”.
- Add more advanced CNN architectures (e.g., LeNet, ResNet).
- Extend the interface to save custom drawings and re-train with them.
- Deploy as a simple web app with Flask or FastAPI.

---

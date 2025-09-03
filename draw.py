import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageGrab, ImageOps
import torch
import torch.nn as nn
import numpy as np
from model import CNNModel

model = CNNModel()
model.load_state_dict(torch.load("mnist_model.pth"))
# model.eval()

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST Predictor")
        
        self.canvas = tk.Canvas(master, width=280, height=280, bg='black')
        self.canvas.pack()
        
        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack(side='left')
        
        self.button_clear = tk.Button(master, text="Clear", command=self.clear)
        self.button_clear.pack(side='right')
        
        self.last_x, self.last_y = None, None
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset_last)

    def reset_last(self, event):
        self.last_x, self.last_y = None, None

    
    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill='white', capstyle='round', smooth=True)
        self.last_x, self.last_y = x, y
    
    def clear(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
    
    def predict(self):
        # récupérer l’image du canvas
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        
        # redimensionner à 28x28
        img = img.resize((28,28))
        # inverser couleurs pour MNIST
        img = ImageOps.invert(img)
        # normaliser
        img_array = np.array(img)/255.0 # type: ignore
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.5)/0.5
        
        # prédiction
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        messagebox.showinfo("Prediction", f"Le chiffre prédit est : {prediction}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

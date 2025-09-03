import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageGrab
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import CNNModel

model = CNNModel()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST CNN")
        
        self.canvas = tk.Canvas(master, width=280, height=280, bg='black')
        self.canvas.pack()
        
        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack(side='left')
        
        self.button_clear = tk.Button(master, text="Clear", command=self.clear)
        self.button_clear.pack(side='left')
        
        self.button_debug = tk.Button(master, text="Debug", command=self.debug_visuals)
        self.button_debug.pack(side='right')
        
        self.last_x, self.last_y = None, None
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset_last)

    def reset_last(self, event):
        self.last_x, self.last_y = None, None

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=10, fill='white', capstyle='round', smooth=True
            )
        self.last_x, self.last_y = x, y
    
    def clear(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
    
    def get_processed_image(self):
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        img = img.resize((28,28))
        
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_array, img_tensor
    
    def predict(self):
        _, img_tensor = self.get_processed_image()
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
        messagebox.showinfo("Prediction", f"The number is : {prediction}")
    
    def debug_visuals(self):
        img_array, img_tensor = self.get_processed_image()
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            x = model.conv1(img_tensor)
            x = F.relu(x).squeeze(0)  # shape [32,28,28]
    
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
        
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(img_array, cmap="gray")
        ax1.set_title("Input image (28x28)")
        ax1.axis("off")
        
        ax2 = fig.add_subplot(gs[0,1])
        ax2.bar(range(10), probs)
        ax2.set_xlabel("Digits")
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction probabilities")
        
        num_features = min(32, x.shape[0])
        cols = 8
        rows = int(np.ceil(num_features / cols))

        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_title("First Conv Layer Feature Maps")
        ax3.axis("off")

        for i in range(num_features):
            row_idx = i // cols
            col_idx = i % cols
            ax_inset = ax3.inset_axes((
                col_idx / cols,              # x0
                1 - (row_idx + 1) / rows,    # y0
                1 / cols,                    # width
                1 / rows                     # height
            ))
            ax_inset.imshow(x[i].cpu(), cmap="gray")
            ax_inset.axis("off")

        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

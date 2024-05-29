#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import joblib

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the classes
classes = ['Guest', 'Member']

# Load the trained data
if os.path.exists("trained_data.pkl"):
    X, y = joblib.load("trained_data.pkl")
else:
    X, y = np.array([]), []

# Train the k-NN model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

def detect_and_predict_class(image_path):
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", "Unable to read the image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(detected_faces) != 1:
        messagebox.showwarning("Warning", "Please select an image with exactly one face.")
        return

    x, y, w, h = detected_faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)

    # Predict the class of the closest face using k-NN
    predicted_class = knn.predict(face)

    return predicted_class[0]  # Return the predicted class

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        predicted_class = detect_and_predict_class(file_path)
        print("Shape of X:", X.shape)
        print("Shape of y:", len(y))
        print("Classes:", y)

        if predicted_class:
            messagebox.showinfo("Prediction", f"The predicted class for this person is: {predicted_class}")
        else:
            messagebox.showwarning("Warning", "No face detected or multiple faces detected.")

# Create GUI
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("500x350")

main_frame = tk.Frame(root, bg="#F0F0F0")
main_frame.pack(expand=True, fill="both")

# Logo
logo_frame = tk.Frame(main_frame, bg="#F0F0F0")
logo_frame.pack(pady=10)

logo_image = Image.open("src/icon.png")
logo_image = logo_image.resize((100, 100))
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(logo_frame, image=logo_photo)
logo_label.image = logo_photo
logo_label.pack()

# Program Description
description_label = tk.Label(main_frame, text="Description: This program detects faces and predicts their classes.", bg="#F0F0F0", font=("Arial", 12))
description_label.pack(pady=10)

# File Chooser Frame
file_chooser_frame = tk.Frame(main_frame, bg="#F0F0F0")
file_chooser_frame.pack(pady=10, side=tk.TOP)

file_chooser_button = tk.Button(file_chooser_frame, text="Browse Image", command=browse_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
file_chooser_button.pack()

root.mainloop()


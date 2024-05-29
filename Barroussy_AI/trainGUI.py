#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox
from PIL import Image, ImageTk

import cv2
import os
import numpy as np
import joblib

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the classes
classes = ['Guest', 'Member']

def train_faces():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    print("Selected file paths:", file_paths)  # Debugging: Print selected file paths
    if file_paths:
        class_name = class_name_var.get()
        if class_name:
            num_images = len(file_paths)
            success_message = f"Class: {class_name}\nNumber of images given: {num_images}\n"
            success_count = 0
            
            for file_path in file_paths:
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        print("Error: Unable to read the image:", file_path)
                        continue  # Skip to the next image if unable to read
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                    
                    if len(detected_faces) == 1:  # Ensure only one face is detected
                        x, y, w, h = detected_faces[0]
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (100, 100))  # Resize the face to the training size
                        face = face.flatten().reshape(1, -1)  # Flatten the face for the model

                        # Load existing data or create new data
                        if os.path.exists("trained_data.pkl"):
                            X, y = joblib.load("trained_data.pkl")
                            X = np.concatenate((X, face), axis=0)
                            y.append(class_name)
                        else:
                            X = face
                            y = [class_name]

                        # Save the updated data
                        joblib.dump((X, y), "trained_data.pkl")

                        success_count += 1
                    else:
                        messagebox.showwarning("Warning", f"Skipping image {file_path}. Please select an image with exactly one face.")

                except Exception as e:
                    print("An error occurred while processing image:", file_path, "-", e)
                    
            success_message += f"Images saved: {success_count}"
            print(success_message)
            messagebox.showinfo("Success", "Images saved successfully!")
        else:
            messagebox.showwarning("Warning", "Please select a class.")
    else:
        messagebox.showwarning("Warning", "No files selected.")

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
description_label = tk.Label(main_frame, text="Description: This program trains faces for recognition.", bg="#F0F0F0", font=("Arial", 12))
description_label.pack(pady=10)

# Class Selection Frame
class_label_text = "Select The Reference: "
class_selection_frame = tk.Frame(main_frame, bg="#F0F0F0")
class_selection_frame.pack(pady=10, side=tk.TOP)

class_label = tk.Label(class_selection_frame, text=class_label_text, bg="#F0F0F0", font=("Arial", 10,"bold"))
class_label.pack(side=tk.LEFT)

class_name_var = tk.StringVar()
class_combobox = Combobox(class_selection_frame, textvariable=class_name_var, values=classes, state='readonly')
class_combobox.pack(padx=10, pady=5, side=tk.LEFT)

# File Chooser Frame
file_chooser_frame = tk.Frame(main_frame, bg="#F0F0F0")
file_chooser_frame.pack(pady=10, side=tk.TOP)

file_chooser_button = tk.Button(file_chooser_frame, text="Choose Files", command=train_faces, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
file_chooser_button.pack()

root.mainloop()


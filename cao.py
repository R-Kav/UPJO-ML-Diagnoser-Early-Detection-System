import os
import numpy as np
from skimage import io, color, feature
import os
import tkinter as tk
from tkinter import Label, Entry, Button
from PIL import Image, ImageTk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score

# Ground truth labels and predicted labels for UPJO detection
true_labels = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
predicted_labels = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]

# Initialize a list to store precision values at each step
precision_values = []

for i in range(1, len(true_labels) + 1):
    subset_true = true_labels[:i]
    subset_predicted = predicted_labels[:i]
    
    # Set zero_division to control behavior
    precision = precision_score(subset_true, subset_predicted, zero_division=1)
    precision_values.append(precision)



# Data features and labels
data = {
    'age': np.random.randint(18, 60, size=100),
    'gender': np.random.choice(['Male', 'Female'], size=100),
    'ureter_diameter_mm': np.random.uniform(2, 10, size=100),
    'pain_level': np.random.randint(0, 10, size=100),
    'kidney_size_cm': np.random.uniform(8, 16, size=100),
    'family_history': np.random.choice([0, 1], size=100)
}
labels = np.random.choice([0, 1], size=100)

# Convert the data dictionary to a structured array
non_image_data = np.array([(data['age'][i], data['ureter_diameter_mm'][i], data['pain_level'][i], data['kidney_size_cm'][i], data['family_history'][i], labels[i]) for i in range(len(labels))],
                           dtype=[('age', int), ('ureter_diameter_mm', float), ('pain_level', int), ('kidney_size_cm', float), ('family_history', int), ('label', int)])

# Split the data into train and test sets
X = np.array([(row['age'], row['ureter_diameter_mm'], row['pain_level'], row['kidney_size_cm'], row['family_history']) for row in non_image_data])
y = np.array([row['label'] for row in non_image_data])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Intercept and coefficients
coefficients = model.coef_


# Make predictions on the test set
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate the log loss
loss = log_loss(y_test, y_prob)

# Function to update the info label with user input
def update_info():
    user_info = user_info_entry.get()
    info_label.config(text=f"User Info: {user_info}")

    # Get the diagnosis result
    image_path = os.path.join('C:/Users/admin/Downloads', 'ultrasound_image1.jpg')
    result = classify_upjo(image_path)
    diagnosis_label.config(text=f"Diagnosis: {result}")

# Function to classify UPJO based on image properties
def classify_upjo(image_path):
    try:
        # Load and preprocess the image
        img = io.imread(image_path)
        img_gray = color.rgb2gray(img)  # Convert to grayscale

        # Calculate image properties
        mean_intensity = np.mean(img_gray)
        edges = feature.canny(img_gray)
        edge_density = np.sum(edges) / edges.size
        
        # criteria to classify UPJO based on image properties
        if mean_intensity < 0.5 and edge_density > 0.1:
            return "UPJO Detected"
        else:
            return "No UPJO Detected"
    except Exception as e:
        return f"Error: {str(e)}"

# Create a main window
root = tk.Tk()
root.title("Early detection of UPJO using Machine Learning")

# User info entry
user_info_label = Label(root, text="Enter User Info:")
user_info_label.pack()

user_info_entry = Entry(root)
user_info_entry.pack()

# Load and display an image
image_path = os.path.join('C:/Users/admin/Downloads', 'ultrasound_image2.jpg')
image = Image.open(image_path)  # Replace with your image path
photo = ImageTk.PhotoImage(image)

image_label = Label(root, image=photo)
image_label.pack()

# Information label
info_text = "Enter Patient information to get result"
info_label = Label(root, text=info_text)
info_label.pack()

# Button to update user info and get diagnosis
update_button = Button(root, text="Update Info and Get Diagnosis", command=update_info)
update_button.pack()

# Diagnosis label
diagnosis_label = Label(root, text="Diagnosis: (No result yet)")
diagnosis_label.pack()

# Run the Tkinter main loop
root.mainloop()
print("Precision values at each step:")
print(precision_values)
print(f'Loss Values at step: {coefficients}')

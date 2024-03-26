# UPJO-ML-Diagnoser-Early-Detection-System
This Python script demonstrates a system for early detection of Ureteropelvic Junction Obstruction (UPJO) using machine learning and image processing techniques. Here's a brief explanation of the script:

Ground Truth Evaluation: The script begins by evaluating the performance of a model that predicts UPJO based on ground truth labels and predicted labels. It computes precision values at each step of prediction using the precision_score function from scikit-learn.

Non-Image Data Processing: Next, it generates sample non-image data features and labels for training a logistic regression model. The features include attributes like age, gender, ureter diameter, pain level, kidney size, and family history. <br /

Model Training and Evaluation: It splits the non-image data into training and testing sets, trains a logistic regression model on the training data, and evaluates the model's performance on the test set using log loss.

User Interface Setup: The script sets up a simple Tkinter-based GUI for user interaction. Users can input their information, and upon clicking a button, the system provides a diagnosis based on both the user's input and an ultrasound image of the patient.

Image Classification: The classify_upjo function processes the ultrasound image to extract relevant properties such as mean intensity and edge density. Based on these properties, it determines whether UPJO is detected or not.

GUI Elements: The GUI elements include an entry widget for user information, an image display area, a label to show information input by the user, a button to trigger diagnosis, and a label to display the diagnosis result.

Main Loop: The script runs the Tkinter main loop, which handles user interactions and GUI updates.

Output: After running the GUI, the script prints the precision values at each step of the evaluation process and the coefficients of the logistic regression model.

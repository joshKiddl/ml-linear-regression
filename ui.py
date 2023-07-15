import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Create a Tkinter window
window = tk.Tk()
window.title("Linear Regression Model")
window.geometry("300x200")

# Function to calculate and display the predicted salary
def predict_salary():
    try:
        age = float(age_entry.get())
        weight = float(weight_entry.get())

        # Make the prediction
        input_data = np.array([[age, weight]])
        predicted_salary = model.predict(input_data)

        # Display the predicted salary
        messagebox.showinfo("Prediction", f"The predicted salary is: {predicted_salary[0]:.2f}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values for Age and Weight.")

# Create labels and entry fields
age_label = tk.Label(window, text="Age:")
age_label.pack()
age_entry = tk.Entry(window)
age_entry.pack()

weight_label = tk.Label(window, text="Weight:")
weight_label.pack()
weight_entry = tk.Entry(window)
weight_entry.pack()

predict_button = tk.Button(window, text="Predict Salary", command=predict_salary)
predict_button.pack()

# Run the Tkinter event loop
window.mainloop()

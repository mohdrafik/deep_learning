import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# 1. Define your Softmax Function
def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution. """    
    a = np.zeros(np.shape(z)[0])
    for i in range(np.shape(z)[0]):
        a[i] = np.exp(z[i]) / np.sum(np.exp(z)) 
    return a

# 2. Define the plotting function with STRICT parameters (No **kwargs here)
def interactive_softmax_plot(z_init, softmax_func):
    """
    Creates an interactive side-by-side horizontal bar chart for logits and softmax.
    """
    num_classes = len(z_init)
    categories = [f"Class {i}" for i in range(num_classes)]
    
    # Create the update function that the sliders will trigger
    def update_plot(**kwargs):
        # Extract the current slider values into a numpy array
        z_current = np.array([kwargs[f'z_{i}'] for i in range(num_classes)])
        
        # Call the math function correctly to get an array of numbers
        a_current = softmax_func(z_current)
        
        # Clear previous plot and set up the figure
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3))
        
        # Left Plot: Raw Logits (z)
        ax1.barh(categories, z_current, color='royalblue')
        ax1.set_title("Raw Logits (z)")
        ax1.set_xlim(-10, 10) 
        ax1.axvline(0, color='black', linewidth=0.8) 
        ax1.invert_yaxis()    
        
        # Right Plot: Softmax Probabilities (a)
        ax2.barh(categories, a_current, color='darkorange')
        ax2.set_title("Softmax Probabilities (a=my_softmax(z))")
        ax2.set_xlim(0, 1)    
        ax2.invert_yaxis()
        
        plt.tight_layout(pad=0.3)
        plt.show()

    # 3. Generate a slider for every value in the z array
    sliders = {
        f'z_{i}': widgets.FloatSlider(
            value=z_init[i], 
            min=-10.0, # Kept at -10 so you can see negative logits!
            max=10.0, 
            step=0.1, 
            description=f'z[{i}]:'
        ) 
        for i in range(num_classes)
    }

    # Connect the sliders to the update function
    out = widgets.interactive_output(update_plot, sliders)
    
    # Display the sliders in neat rows of 2
    slider_list = list(sliders.values())
    rows = [widgets.HBox(slider_list[i:i+2]) for i in range(0, len(slider_list), 2)]
    ui = widgets.VBox(rows)
    
    display(ui, out)
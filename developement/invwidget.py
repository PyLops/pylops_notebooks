import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from ipywidgets import interactive_output, IntSlider, VBox, HBox, Text, Button, HTML
from IPython.display import display

def display_inversion(data, model_history, cost, error=None, model_true=None, iteration=None):
    if iteration is None:
        iteration = len(model_history)-1
    npanels = 3 if model_true is None else 5
    npanels = npanels if error is None else npanels + 1
    mtrue_panel = 0 if model_true is None else 1
    mtrue_panel = mtrue_panel if error is None else mtrue_panel + 1

    vmin, vmax = -np.abs(data.max()), np.abs(data.max())
    fig, axs = plt.subplots(1, npanels, figsize=(3.5*npanels, 6))
    # Data
    axs[0].imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].axis('tight')
    axs[0].set_title('Data')
    
    # Inverted model
    axs[1].imshow(model_history[iteration], cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].axis('tight')
    axs[1].set_title('Inverted Model')

    if model_true is not None:
        axs[2].imshow(model_true, cmap='gray', vmin=vmin, vmax=vmax)
        axs[2].set_title('True Model')
        axs[2].axis('tight')
        axs[3].imshow(model_true - model_history[iteration], cmap='gray', vmin=vmin, vmax=vmax)
        axs[3].set_title('Model Error')
        axs[3].axis('tight')

    # Residual Norm
    axs[2+mtrue_panel].plot(cost[:iteration], 'k', lw=2)
    axs[2+mtrue_panel].set_xlim(0, len(model_history))
    axs[2+mtrue_panel].set_ylim(cost.min(), cost.max())
    axs[2+mtrue_panel].set_title('Residual Norm')
    
    if error is not None:
        axs[3+mtrue_panel].plot(error[:iteration], 'k', lw=2)
        axs[3+mtrue_panel].set_xlim(0, len(model_history))
        axs[3+mtrue_panel].set_ylim(error.min(), error.max())
        axs[3+mtrue_panel].set_title('Error Norm')
        
        
# Create widget
def inversion_widget(data, model_history, cost, error=None, model_true=None, title='Inversion Widget'):
    niters = len(model_history)
    curr_iter = niters - 1
    title = "<b>%s</b>" % title
    slider_iters = IntSlider(min=1, max=niters-1, value=curr_iter, step=1, description='Iteration')
    title = HTML(value=title, placeholder="Inversion Widget", description='')

    def handle_iters_change(change):
        global curr_iter
        curr_iter = change.new

    slider_iters.observe(handle_iters_change, names='value')

    out = interactive_output(lambda iteration: display_inversion(data, model_history, cost, 
                                                                 error=error, model_true=model_true,
                                                                 iteration=iteration),
                             {"iteration": slider_iters})
    ui = HBox([title, slider_iters])

    display(ui, out, continuous_update=False)
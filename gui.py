"""
GUI for The Neural Project - Interactive Neural Network Interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from neural_core.network import NeuralNetwork
from adapters import DataHandler


class NeuralProjectGUI:
    """Main GUI window for The Neural Project"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("The Neural Project - Interactive Neural Network")
        self.root.geometry("1200x800")
        
        self.network = None
        self.training = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Network Configuration
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Network Configuration")
        self._create_config_tab()
        
        # Tab 2: Training
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Training")
        self._create_training_tab()
        
        # Tab 3: Visualization
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        self._create_visualization_tab()
        
        # Tab 4: Memory & Info
        self.info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.info_frame, text="Network Info")
        self._create_info_tab()
    
    def _create_config_tab(self):
        """Create network configuration tab"""
        # Left panel - Configuration
        left_panel = ttk.LabelFrame(self.config_frame, text="Network Architecture", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input size
        ttk.Label(left_panel, text="Input Features:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_size_var = tk.IntVar(value=10)
        ttk.Spinbox(left_panel, from_=1, to=100, textvariable=self.input_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Hidden layers
        ttk.Label(left_panel, text="Hidden Layer 1:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hidden1_var = tk.IntVar(value=32)
        ttk.Spinbox(left_panel, from_=4, to=256, textvariable=self.hidden1_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(left_panel, text="Hidden Layer 2:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.hidden2_var = tk.IntVar(value=16)
        ttk.Spinbox(left_panel, from_=4, to=256, textvariable=self.hidden2_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Output size
        ttk.Label(left_panel, text="Output Classes:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_size_var = tk.IntVar(value=5)
        ttk.Spinbox(left_panel, from_=2, to=20, textvariable=self.output_size_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Activation function
        ttk.Label(left_panel, text="Activation:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.activation_var = tk.StringVar(value="relu")
        ttk.Combobox(left_panel, textvariable=self.activation_var, 
                     values=["relu", "sigmoid", "tanh"], state="readonly", width=8).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Create button
        ttk.Button(left_panel, text="Create Network", command=self._create_network).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=20)
        
        # Right panel - Data
        right_panel = ttk.LabelFrame(self.config_frame, text="Data Setup", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(right_panel, text="Data Type:").pack(anchor=tk.W, pady=5)
        self.data_type_var = tk.StringVar(value="synthetic")
        ttk.Combobox(right_panel, textvariable=self.data_type_var,
                     values=["synthetic", "load_from_file"], state="readonly").pack(fill=tk.X, pady=5)
        
        # Synthetic data options
        ttk.Label(right_panel, text="Number of Samples:").pack(anchor=tk.W, pady=5)
        self.num_samples_var = tk.IntVar(value=200)
        ttk.Spinbox(right_panel, from_=50, to=1000, textvariable=self.num_samples_var, width=20).pack(fill=tk.X, pady=5)
        
        ttk.Label(right_panel, text="Train/Test Split:").pack(anchor=tk.W, pady=5)
        self.train_ratio_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(right_panel, from_=0.5, to=0.95, increment=0.05, textvariable=self.train_ratio_var, width=20).pack(fill=tk.X, pady=5)
        
        # Generate button
        ttk.Button(right_panel, text="Generate Data", command=self._generate_data).pack(fill=tk.X, pady=20)
        
        # Status
        self.config_status = tk.StringVar(value="Ready to create network")
        ttk.Label(right_panel, textvariable=self.config_status, foreground="blue").pack(anchor=tk.W, pady=10)
    
    def _create_training_tab(self):
        """Create training tab"""
        # Control panel
        control_panel = ttk.LabelFrame(self.training_frame, text="Training Controls", padding=10)
        control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Hyperparameters
        col1 = ttk.Frame(control_panel)
        col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(col1, text="Epochs:").pack(anchor=tk.W)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(col1, from_=10, to=1000, textvariable=self.epochs_var, width=15).pack(fill=tk.X, pady=5)
        
        ttk.Label(col1, text="Batch Size:").pack(anchor=tk.W)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(col1, from_=4, to=128, textvariable=self.batch_size_var, width=15).pack(fill=tk.X, pady=5)
        
        # Training buttons
        button_panel = ttk.Frame(control_panel)
        button_panel.pack(side=tk.RIGHT, padx=5)
        
        self.train_button = ttk.Button(button_panel, text="Start Training", command=self._start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_panel, text="Stop", command=self._stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress panel
        progress_panel = ttk.LabelFrame(self.training_frame, text="Training Progress", padding=10)
        progress_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress bar
        ttk.Label(progress_panel, text="Progress:").pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Loss display
        ttk.Label(progress_panel, text="Current Loss:").pack(anchor=tk.W)
        self.loss_display = tk.Label(progress_panel, text="N/A", font=("Arial", 12), foreground="red")
        self.loss_display.pack(anchor=tk.W, pady=5)
        
        # Accuracy display
        ttk.Label(progress_panel, text="Best Accuracy:").pack(anchor=tk.W)
        self.accuracy_display = tk.Label(progress_panel, text="N/A", font=("Arial", 12), foreground="green")
        self.accuracy_display.pack(anchor=tk.W, pady=5)
        
        # Training status
        self.training_status = tk.StringVar(value="Ready to train")
        ttk.Label(progress_panel, textvariable=self.training_status, foreground="blue").pack(anchor=tk.W, pady=10)
    
    def _create_visualization_tab(self):
        """Create visualization tab"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        button_panel = ttk.Frame(self.viz_frame)
        button_panel.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_panel, text="Plot Loss", command=self._plot_loss).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_panel, text="Plot Accuracy", command=self._plot_accuracy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_panel, text="Clear", command=self._clear_plot).pack(side=tk.LEFT, padx=5)
    
    def _create_info_tab(self):
        """Create network info tab"""
        info_panel = ttk.Frame(self.info_frame)
        info_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Network info display
        self.info_text = tk.Text(info_panel, height=20, width=80, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Button to refresh
        ttk.Button(self.info_frame, text="Refresh Info", command=self._refresh_info).pack(pady=5)
    
    def _create_network(self):
        """Create neural network with current parameters"""
        try:
            self.network = NeuralNetwork(
                input_size=self.input_size_var.get(),
                hidden_layers=[self.hidden1_var.get(), self.hidden2_var.get()],
                output_size=self.output_size_var.get(),
                activation=self.activation_var.get()
            )
            
            info = self.network.get_network_info()
            self.config_status.set(
                f"Network created! Layers: {info['layer_sizes']}, Total neurons: {info['total_neurons']}"
            )
            self._refresh_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create network: {str(e)}")
    
    def _generate_data(self):
        """Generate synthetic training data"""
        try:
            num_samples = self.num_samples_var.get()
            output_size = self.output_size_var.get()
            input_size = self.input_size_var.get()
            train_ratio = self.train_ratio_var.get()
            
            # Generate synthetic data
            np.random.seed(42)
            X = np.random.randn(num_samples, input_size)
            y = np.eye(output_size)[np.random.randint(0, output_size, num_samples)]
            
            # Normalize
            X = DataHandler.normalize(X)
            
            # Split
            self.X_train, self.X_test, self.y_train, self.y_test = DataHandler.split_data(X, y, train_ratio)
            
            self.config_status.set(
                f"Data generated! Train: {len(self.X_train)}, Test: {len(self.X_test)}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
    
    def _start_training(self):
        """Start training in a separate thread"""
        if self.network is None:
            messagebox.showwarning("Warning", "Please create a network first")
            return
        
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please generate data first")
            return
        
        self.training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start training in separate thread
        thread = threading.Thread(target=self._training_thread)
        thread.daemon = True
        thread.start()
    
    def _training_thread(self):
        """Training thread"""
        try:
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            
            for epoch in range(epochs):
                if not self.training:
                    break
                
                # Train for one epoch
                self.network.train(self.X_train, self.y_train, epochs=1, batch_size=batch_size)
                
                # Evaluate
                test_loss, test_accuracy = self.network.evaluate(self.X_test, self.y_test)
                
                # Update UI
                self.progress_var.set((epoch + 1) / epochs * 100)
                self.loss_display.config(text=f"{self.network.loss_history[-1]:.6f}")
                self.accuracy_display.config(text=f"{test_accuracy:.4f}")
                self.training_status.set(f"Epoch {epoch + 1}/{epochs}")
                
                self.root.update()
            
            if self.training:
                self.training_status.set("Training complete!")
                messagebox.showinfo("Success", "Training finished successfully!")
        
        except Exception as e:
            self.training_status.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        
        finally:
            self.training = False
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def _stop_training(self):
        """Stop training"""
        self.training = False
        self.training_status.set("Training stopped")
    
    def _plot_loss(self):
        """Plot training loss"""
        if self.network is None or not self.network.loss_history:
            messagebox.showwarning("Warning", "No training history available")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.network.loss_history, label="Training Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def _plot_accuracy(self):
        """Plot accuracy"""
        if self.network is None or not self.network.accuracy_history:
            messagebox.showwarning("Warning", "No accuracy history available")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.network.accuracy_history, label="Test Accuracy", linewidth=2, color="green")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Over Evaluations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def _clear_plot(self):
        """Clear the plot"""
        self.fig.clear()
        self.canvas.draw()
    
    def _refresh_info(self):
        """Refresh network information display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.network is None:
            self.info_text.insert(tk.END, "No network created yet. Create one in the Configuration tab.")
        else:
            info = self.network.get_network_info()
            text = f"""
NETWORK INFORMATION
{'='*50}

Architecture:
  Input Features:     {info['input_size']}
  Hidden Layers:      {info['layer_sizes'][1:-1]}
  Output Classes:     {info['output_size']}
  Total Layers:       {info['num_layers']}

Capacity:
  Total Neurons:      {info['total_neurons']}
  Growth Events:      {info['growth_events']}

Performance:
  Current Loss:       {info['current_loss'] if info['current_loss'] else 'N/A'}
  Best Accuracy:      {info['best_accuracy'] if info['best_accuracy'] else 'N/A'}

Memory Systems:
  Short-term Memory:  {len(self.network.memory.short_term.recall())} items
  Long-term Patterns: {len(self.network.memory.long_term.patterns)} stored
  Episodic Memories:  {len(self.network.memory.episodic.episodes)} episodes

Learning:
  Current LR:         {self.network.adaptive_learning.learning_rate:.6f}
  Loss History:       {len(self.network.loss_history)} epochs tracked
  Accuracy History:   {len(self.network.accuracy_history)} evaluations
"""
            self.info_text.insert(tk.END, text)
        
        self.info_text.config(state=tk.DISABLED)


def main():
    """Launch the GUI"""
    root = tk.Tk()
    app = NeuralProjectGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

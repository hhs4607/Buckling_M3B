This file has been removed as per project rules to avoid duplication.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI for Buckling Analysis M3
-------------------------------------
Professional GUI interface for composite beam buckling analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import the analysis functions
from buckling_analysis_M3a import (
    eval_m2_Pcr,
    eval_m3_Pcr_and_mode,
    koiter_curves_from_mode,
    plot_load_deflection
)


class BucklingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Composite Beam Buckling Analysis - M3")
        self.root.geometry("1200x800")

        # Storage for results
        self.current_results = None
        self.is_running = False

        # Create main layout
        self.create_layout()
        self.load_defaults()

    def create_layout(self):
        """Create the main GUI layout"""

        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Inputs (scrollable)
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create canvas for scrolling
        canvas = tk.Canvas(left_frame, width=450)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Right panel - Results
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create input sections
        self.create_input_sections()

        # Create results section
        self.create_results_section(right_frame)

        # Bind mousewheel to canvas
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def create_input_sections(self):
        """Create all input parameter sections"""

        # Dictionary to store all entry widgets
        self.entries = {}

        row = 0

        # ========== GEOMETRY SECTION ==========
        row = self.create_section_header("GEOMETRY", row)

        params = [
            ("L", "Beam Length (m)", 1.5),
            ("b_root", "Root Width (m)", 0.08),
            ("b_tip", "Tip Width (m)", 0.04),
            ("h_root", "Root Core Height (m)", 0.025),
            ("h_tip", "Tip Core Height (m)", 0.015),
            ("w_f", "Flange Width (m)", 0.02),
        ]

        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ========== FACE LAMINATE SECTION ==========
        row = self.create_section_header("FACE LAMINATE", row)

        params = [
            ("t_face_total", "Total Face Thickness (m)", 0.002),
            ("face_angles", "Face Ply Angles (deg)", "0,45,-45,90"),
        ]

        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ========== WEB LAMINATE SECTION ==========
        row = self.create_section_header("WEB LAMINATE", row)

        params = [
            ("t_web_total", "Total Web Thickness (m)", 0.0015),
            ("web_angles", "Web Ply Angles (deg)", "0,90"),
        ]

        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ========== MATERIAL PROPERTIES SECTION ==========
        row = self.create_section_header("MATERIAL PROPERTIES", row)

        params = [
            ("Ef", "Fiber Modulus (Pa)", 230e9),
            ("Em", "Matrix Modulus (Pa)", 3.5e9),
            ("Gf", "Fiber Shear Modulus (Pa)", 90e9),
            ("nuf", "Fiber Poisson's Ratio", 0.2),
            ("num", "Matrix Poisson's Ratio", 0.35),
            ("Vf", "Fiber Volume Fraction", 0.6),
        ]

        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ========== BOUNDARY CONDITIONS SECTION ==========
        row = self.create_section_header("BOUNDARY CONDITIONS", row)

        params = [
            ("Ktheta_root_per_m", "Root Spring Stiffness (N·m/m)", 1e9),
        ]

        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ========== ANALYSIS SETTINGS SECTION ==========
        row = self.create_section_header("ANALYSIS SETTINGS", row)

        # Solver core selection
        core_frame = ttk.Frame(self.scrollable_frame)
        core_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=10)

        ttk.Label(core_frame, text="Solver Core:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.core_var = tk.StringVar(value="m3")
        ttk.Radiobutton(core_frame, text="M3 (Two-term + Spring)", variable=self.core_var, value="m3").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(core_frame, text="M2 (Fast, Clamped)", variable=self.core_var, value="m2").pack(side=tk.LEFT, padx=5)

        row += 1

        # Numerical parameters
        params = [
            ("PPW", "Points Per Wavelength", 60),
            ("nx_min", "Minimum Grid Points", 1801),
        ]

        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ========== ACTION BUTTONS ==========
        row += 1
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)

        # Run button
        self.run_button = ttk.Button(button_frame, text="▶ RUN ANALYSIS", command=self.run_analysis, width=20)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Load/Save buttons
        ttk.Button(button_frame, text="📁 Load Config", command=self.load_config, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Save Config", command=self.save_config, width=15).pack(side=tk.LEFT, padx=5)

        # Progress bar
        row += 1
        self.progress = ttk.Progressbar(self.scrollable_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=10)

        # Status label
        row += 1
        self.status_label = ttk.Label(self.scrollable_frame, text="Ready", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=2, pady=5)

    def create_section_header(self, title, row):
        """Create a section header"""
        frame = ttk.Frame(self.scrollable_frame)
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 5), padx=5)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=(0, 5))
        ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack(anchor='w')

        return row + 1

    def create_input_row(self, key, label, default, row):
        """Create a single input row"""
        # Label
        ttk.Label(self.scrollable_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=10, pady=3)

        # Entry
        entry = ttk.Entry(self.scrollable_frame, width=25)
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=3)
        entry.insert(0, str(default))

        self.entries[key] = entry

        return row + 1

    def create_results_section(self, parent):
        """Create results display section"""

        # Title
        title_label = ttk.Label(parent, text="RESULTS", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        # Results text area
        text_frame = ttk.LabelFrame(parent, text="Analysis Summary", padding="10")
        text_frame.pack(fill=tk.BOTH, pady=(0, 10))

        self.results_text = tk.Text(text_frame, height=10, width=50, font=('Courier', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for text
        text_scroll = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=text_scroll.set)

        # Plot area
        plot_frame = ttk.LabelFrame(parent, text="Load-Deflection Curve", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Tip Deflection δ [m]")
        self.ax.set_ylabel("Load P [N]")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.text(0.5, 0.5, 'Run analysis to see results',
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12, alpha=0.5)

        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Export button
        export_frame = ttk.Frame(parent)
        export_frame.pack(pady=10)

        ttk.Button(export_frame, text="📊 Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="📄 Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)

    def get_values(self):
        """Extract all values from entry widgets"""
        vals = {}

        # Get numeric values
        numeric_keys = ["L", "b_root", "b_tip", "h_root", "h_tip", "w_f",
                       "t_face_total", "t_web_total", "Ef", "Em", "Gf",
                       "nuf", "num", "Vf", "Ktheta_root_per_m", "PPW", "nx_min"]

        for key in numeric_keys:
            try:
                val_str = self.entries[key].get().strip()
                vals[key] = float(val_str)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {key}: {val_str}")
                return None

        # Get string values (angles)
        vals["face_angles"] = self.entries["face_angles"].get().strip()
        vals["web_angles"] = self.entries["web_angles"].get().strip()

        return vals

    def validate_inputs(self, vals):
        """Validate input values"""
        errors = []

        if vals["L"] <= 0:
            errors.append("Length must be positive")
        if vals["b_root"] <= 0 or vals["b_tip"] <= 0:
            errors.append("Widths must be positive")
        if vals["h_root"] <= 0 or vals["h_tip"] <= 0:
            errors.append("Heights must be positive")
        if not (0 < vals["Vf"] < 1):
            errors.append("Fiber volume fraction must be between 0 and 1")
        if vals["Ef"] <= 0 or vals["Em"] <= 0:
            errors.append("Moduli must be positive")
        if vals["Ktheta_root_per_m"] < 0:
            errors.append("Spring stiffness must be non-negative")

        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False

        return True

    def run_analysis(self):
        """Run the buckling analysis"""
        if self.is_running:
            messagebox.showwarning("Warning", "Analysis already running!")
            return

        # Get values
        vals = self.get_values()
        if vals is None:
            return

        # Validate
        if not self.validate_inputs(vals):
            return

        # Start analysis in thread
        self.is_running = True
        self.run_button.config(state='disabled')
        self.progress.start(10)
        self.status_label.config(text="Running analysis...", foreground="orange")

        thread = threading.Thread(target=self.analysis_worker, args=(vals,))
        thread.start()

    def analysis_worker(self, vals):
        """Worker thread for analysis"""
        try:
            core = self.core_var.get()
            PPW = int(vals["PPW"])
            nx_min = int(vals["nx_min"])

            if core == "m2":
                # M2 analysis (fast)
                Pcr = eval_m2_Pcr(vals, PPW=PPW, nx_min=nx_min)
                results = {
                    "core": "M2",
                    "Pcr": Pcr,
                    "mode_available": False
                }
            else:
                # M3 analysis (full)
                mode = eval_m3_Pcr_and_mode(vals, PPW=PPW, nx_min=nx_min, return_mode=True)
                P, dlin, dloc, dtot, dcr, more = koiter_curves_from_mode(mode)

                results = {
                    "core": "M3",
                    "Pcr": mode["Pcr"],
                    "dcr": dcr,
                    "alpha_star": mode["alpha_star"],
                    "beta_star": mode["beta_star"],
                    "lambda_x": mode["lambda_x"],
                    "mode_available": True,
                    "P": P,
                    "dlin": dlin,
                    "dloc": dloc,
                    "dtot": dtot,
                    "mode": mode
                }

            # Update UI in main thread
            self.root.after(0, self.analysis_complete, results)

        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))

    def analysis_complete(self, results):
        """Called when analysis completes successfully"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Analysis complete ✓", foreground="green")

        self.current_results = results

        # Display results
        self.display_results(results)

        # Update plot
        if results["mode_available"]:
            self.plot_results(results)

    def analysis_error(self, error_msg):
        """Called when analysis fails"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Analysis failed ✗", foreground="red")

        messagebox.showerror("Analysis Error", f"Error during analysis:\n{error_msg}")

    def display_results(self, results):
        """Display results in text area"""
        self.results_text.delete(1.0, tk.END)

        text = f"{'='*50}\n"
        text += f"  BUCKLING ANALYSIS RESULTS\n"
        text += f"{'='*50}\n\n"
        text += f"Solver Core: {results['core']}\n"
        text += f"{'-'*50}\n\n"
        text += f"Critical Buckling Load:\n"
        text += f"  Pcr = {results['Pcr']:.2f} N\n"
        text += f"      = {results['Pcr']/1000:.3f} kN\n\n"

        if results["mode_available"]:
            text += f"Critical Deflection:\n"
            text += f"  δcr = {results['dcr']*1000:.4f} mm\n\n"
            text += f"Mode Shape Parameters:\n"
            text += f"  α* = {results['alpha_star']:.4f} 1/m\n"
            text += f"  β* = {results['beta_star']:.4f} 1/m\n"
            text += f"  λx* = {results['lambda_x']:.4f} m\n\n"
        else:
            text += f"\nNote: M2 core provides Pcr only.\n"
            text += f"Use M3 for full mode shapes and curves.\n"

        text += f"{'-'*50}\n"
        text += f"Analysis completed successfully.\n"

        self.results_text.insert(1.0, text)

    def plot_results(self, results):
        """Update plot with results"""
        self.ax.clear()

        P = results["P"]
        dlin = results["dlin"]
        dloc = results["dloc"]
        dtot = results["dtot"]
        Pcr = results["Pcr"]
        dcr = results["dcr"]

        dmax = 1.5 * dcr
        Pmax = 1.5 * Pcr

        # Clip data to plot limits
        def clip(P_arr, d_arr):
            m = (P_arr <= Pmax) & (d_arr <= dmax + 1e-15)
            return d_arr[m], P_arr[m]

        dloc_c, P_loc_c = clip(P, dloc)
        dtot_c, P_tot_c = clip(P, dtot)

        # Linear reference
        fb_fit = float(np.dot(P, dlin) / max(np.dot(P, P), 1e-18)) if P[-1] > 0 else float(dlin[-1] / max(P[-1], 1e-18))
        d_lin_full = np.linspace(0.0, dmax, 400)
        P_lin_full = d_lin_full / max(fb_fit, 1e-18)

        # Plot
        self.ax.plot(d_lin_full, P_lin_full, '--', linewidth=2.0, label='Linear only', color='blue')
        self.ax.plot(dloc_c, P_loc_c, ':', linewidth=2.2, alpha=0.95, label='Nonlinear only', color='orange')
        self.ax.plot(dtot_c, P_tot_c, '-', linewidth=2.4, label='Total', color='green')

        d_at_Pcr = float(np.interp(Pcr, P_tot_c if len(P_tot_c) > 2 else P, dtot_c if len(P_tot_c) > 2 else dtot))
        self.ax.scatter([d_at_Pcr], [Pcr], s=70, edgecolor='k', zorder=4, color='red', label=f'Pcr≈{Pcr:.0f} N')

        self.ax.set_xlabel("Tip Deflection δ [m]")
        self.ax.set_ylabel("Load P [N]")
        self.ax.set_xlim(0, dmax)
        self.ax.set_ylim(0, Pmax)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.legend(loc='lower right', framealpha=0.9)
        self.ax.set_title("Load-Deflection Curve")

        self.fig.tight_layout()
        self.canvas.draw()

    def export_plot(self):
        """Export current plot to file"""
        if self.current_results is None or not self.current_results["mode_available"]:
            messagebox.showwarning("Warning", "No plot available to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to:\n{filename}")

    def export_results(self):
        """Export results to text file"""
        if self.current_results is None:
            messagebox.showwarning("Warning", "No results available to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            content = self.results_text.get(1.0, tk.END)
            with open(filename, 'w') as f:
                f.write(content)
            messagebox.showinfo("Success", f"Results saved to:\n{filename}")

    def load_config(self):
        """Load configuration from JSON file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)

                # Load values into entries
                for key, value in config.items():
                    if key in self.entries:
                        self.entries[key].delete(0, tk.END)
                        self.entries[key].insert(0, str(value))
                    elif key == "core":
                        self.core_var.set(value)

                messagebox.showinfo("Success", "Configuration loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")

    def save_config(self):
        """Save current configuration to JSON file"""
        vals = self.get_values()
        if vals is None:
            return

        vals["core"] = self.core_var.get()

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(vals, f, indent=2)
                messagebox.showinfo("Success", "Configuration saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")

    def load_defaults(self):
        """Load default values (already set in create_input_sections)"""
        pass


def main():
    root = tk.Tk()
    app = BucklingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

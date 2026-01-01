#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI for Buckling Analysis M3
-------------------------------------
Professional GUI interface for composite beam buckling analysis
Supports: FULL (M2/M3), SENS (OAT), SOBOL (UQ)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math

# Import the analysis functions
from buckling_analysis_M3a import (
    eval_m2_Pcr,
    eval_m3_Pcr_and_mode,
    koiter_curves_from_mode,
    plot_contour_on_ax,
    run_sens,
    run_sobol
)

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        self.container = self.scrollable_frame

class BucklingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Composite Beam Buckling Analysis - M3")
        self.root.geometry("1500x900")

        self.is_running = False
        self.temp_excel = Path("temp_gui_config.xlsx").resolve()
        
        self.entries = {}
        self.sens_inputs = {}
        self.sobol_inputs = {}
        
        self.num_params = [
            'L', 'b_root', 'b_tip', 'h_root', 'h_tip', 'w_f', 
            't_face_total', 't_web_total', 
            'Ef', 'Em', 'Gf', 'nuf', 'num', 'Vf', 
            'Ktheta_root_per_m'
        ]

        self.create_layout()

    def create_layout(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Paned Window (Left: Inputs, Right: Tabs)
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Inputs
        left_frame = ttk.Frame(paned, width=400)
        self.create_input_area(left_frame)
        paned.add(left_frame, weight=1)

        # Right panel - Tabs
        right_frame = ttk.Frame(paned)
        self.create_tabs(right_frame)
        paned.add(right_frame, weight=4)

    def create_input_area(self, parent):
        sf = ScrollableFrame(parent)
        sf.pack(fill=tk.BOTH, expand=True)
        
        self.scrollable_input_frame = sf.container
        self.create_input_sections()

    def create_input_sections(self):
        parent = self.scrollable_input_frame
        row = 0
        headers = [
            ("GEOMETRY", [
                ("L", "Beam Length (m)", 1.5),
                ("b_root", "Root Width (m)", 0.08),
                ("b_tip", "Tip Width (m)", 0.04),
                ("h_root", "Root Core Height (m)", 0.025),
                ("h_tip", "Tip Core Height (m)", 0.015),
                ("w_f", "Flange Width (m)", 0.02),
            ]),
            ("FACE LAMINATE", [
                ("t_face_total", "Total Face Thickness (m)", 0.002),
                ("face_angles", "Face Ply Angles (deg)", "0,45,-45,90"),
            ]),
            ("WEB LAMINATE", [
                ("t_web_total", "Total Web Thickness (m)", 0.0015),
                ("web_angles", "Web Ply Angles (deg)", "0,90"),
            ]),
            ("MATERIAL PROPERTIES", [
                ("Ef", "Fiber Modulus (Pa)", 230e9),
                ("Em", "Matrix Modulus (Pa)", 3.5e9),
                ("Gf", "Fiber Shear Modulus (Pa)", 90e9),
                ("nuf", "Fiber Poisson's Ratio", 0.2),
                ("num", "Matrix Poisson's Ratio", 0.35),
                ("Vf", "Fiber Volume Fraction", 0.6),
            ]),
            ("BOUNDARY CONDITIONS", [
                ("Ktheta_root_per_m", "Root Spring (N·m/m)", 1e9),
            ]),
            ("ANALYSIS SETTINGS", [
                ("PPW", "Points Per Wavelength", 60),
                ("nx_min", "Minimum Grid Points", 1801),
            ])
        ]

        for section, params in headers:
            row = self.create_section_header(section, row)
            for key, label, default in params:
                 row = self.create_input_row(key, label, default, row)
        
        # Solver core
        row += 1
        ttk.Label(parent, text="Solver Core:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(10,0), padx=5)
        self.core_var = tk.StringVar(value="m3")
        f = ttk.Frame(parent)
        f.grid(row=row+1, column=0, columnspan=2, sticky=tk.W, padx=5)
        ttk.Radiobutton(f, text="M3 (Two-term + Spring)", variable=self.core_var, value="m3").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f, text="M2 (Fast, Clamped)", variable=self.core_var, value="m2").pack(side=tk.LEFT, padx=5)
        row += 2
        
        # Actions
        f2 = ttk.Frame(parent)
        f2.grid(row=row, column=0, columnspan=2, pady=15)
        ttk.Button(f2, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(f2, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=2)
        row+=1
        
        # Status
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        self.status_label = ttk.Label(parent, text="Ready", foreground="black")
        self.status_label.grid(row=row+1, column=0, columnspan=2)

    def create_section_header(self, title, row):
        parent = self.scrollable_input_frame
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=(10,2))
        ttk.Label(parent, text=title, font=('Arial', 8, 'bold')).grid(row=row+1, column=0, columnspan=2, sticky=tk.W, padx=5)
        return row+2

    def create_input_row(self, key, label, default, row):
        parent = self.scrollable_input_frame
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=10)
        e = ttk.Entry(parent, width=15)
        e.grid(row=row, column=1, sticky=tk.E, padx=10)
        e.insert(0, str(default))
        self.entries[key] = e
        return row+1

    def create_tabs(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_tab_full(self.notebook)
        self.create_tab_sens(self.notebook)
        self.create_tab_sobol(self.notebook)

    # ------------------ FULL ANALYSIS ------------------
    def create_tab_full(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Full Analysis")
        
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, pady=5)
        self.btn_run_full = ttk.Button(top, text="▶ Run Full Analysis", command=self.on_run_full)
        self.btn_run_full.pack(side=tk.LEFT)
        
        self.txt_full = tk.Text(tab, height=6, font=('Courier',10))
        self.txt_full.pack(fill=tk.X, pady=5)
        
        plots = ttk.Frame(tab); plots.pack(fill=tk.BOTH, expand=True)
        plots.columnconfigure(0, weight=1); plots.columnconfigure(1, weight=1); plots.rowconfigure(0, weight=1)
        
        # LD Plot
        p1 = ttk.LabelFrame(plots, text="Load-Deflection")
        p1.grid(row=0, column=0, sticky="nsew", padx=2)
        f1 = Figure(figsize=(4,3), dpi=100)
        self.ax_ld = f1.add_subplot(111)
        self.cv_ld = FigureCanvasTkAgg(f1, master=p1)
        self.cv_ld.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Contour
        p2 = ttk.LabelFrame(plots, text="Mode Contour")
        p2.grid(row=0, column=1, sticky="nsew", padx=2)
        f2 = Figure(figsize=(4,3), dpi=100)
        self.ax_ct = f2.add_subplot(111)
        self.cv_ct = FigureCanvasTkAgg(f2, master=p2)
        self.cv_ct.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------ SENSITIVITY (OAT) ------------------
    def create_tab_sens(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Sensitivity (OAT)")
        
        # Layout: Left Config (Scrollable), Right Plots
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Config
        left = ttk.Frame(paned, width=350)
        paned.add(left, weight=1)
        
        ttk.Label(left, text="Select parameters to vary:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0,5))
        
        # Scrollable list
        sf = ScrollableFrame(left)
        sf.pack(fill=tk.BOTH, expand=True)
        cframe = sf.container
        
        # Headers
        ttk.Label(cframe, text="Param").grid(row=0, column=0, columnspan=2, sticky='w')
        ttk.Label(cframe, text="±%").grid(row=0, column=2, padx=2)
        ttk.Label(cframe, text="Pts").grid(row=0, column=3, padx=2)
        
        r=1
        for p in self.num_params:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(cframe, variable=var)
            chk.grid(row=r, column=0, sticky='w')
            ttk.Label(cframe, text=p).grid(row=r, column=1, sticky='w', padx=2)
            
            e_pct = ttk.Entry(cframe, width=5); e_pct.insert(0,"10")
            e_pct.grid(row=r, column=2, padx=2)
            
            e_pts = ttk.Entry(cframe, width=5); e_pts.insert(0,"5")
            e_pts.grid(row=r, column=3, padx=2)
            
            self.sens_inputs[p] = {"var":var, "pct":e_pct, "pts":e_pts}
            r+=1

        self.btn_run_sens = ttk.Button(left, text="▶ RUN SENSITIVITY", command=self.on_run_sens)
        self.btn_run_sens.pack(pady=10, fill=tk.X)

        # Right: Plots
        right = ttk.Frame(paned)
        paned.add(right, weight=4)
        
        self.fig_sens = Figure(figsize=(6,5), dpi=100)
        self.cv_sens = FigureCanvasTkAgg(self.fig_sens, master=right)
        self.cv_sens.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------ SOBOL (UQ) ------------------
    def create_tab_sobol(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Uncertainty (Sobol)")
        
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left = ttk.Frame(paned, width=350)
        paned.add(left, weight=1)
        
        # Top Settings
        sett = ttk.LabelFrame(left, text="Sobol Settings")
        sett.pack(fill=tk.X, pady=5)
        ttk.Label(sett, text="N_base:").grid(row=0, column=0, padx=5)
        self.ent_sobol_n = ttk.Entry(sett, width=8); self.ent_sobol_n.insert(0,"128")
        self.ent_sobol_n.grid(row=0, column=1)
        
        ttk.Label(left, text="Uncertain Parameters:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10,5))
        
        sf = ScrollableFrame(left)
        sf.pack(fill=tk.BOTH, expand=True)
        cframe = sf.container
        
        ttk.Label(cframe, text="Param").grid(row=0, column=0, columnspan=2, sticky='w')
        ttk.Label(cframe, text="Low").grid(row=0, column=2)
        ttk.Label(cframe, text="High").grid(row=0, column=3)
        
        r=1
        for p in self.num_params:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(cframe, variable=var)
            chk.grid(row=r, column=0, sticky='w')
            ttk.Label(cframe, text=p).grid(row=r, column=1, sticky='w', padx=2)
            
            e_lo = ttk.Entry(cframe, width=8)
            e_lo.grid(row=r, column=2, padx=2)
            e_hi = ttk.Entry(cframe, width=8)
            e_hi.grid(row=r, column=3, padx=2)
            
            self.sobol_inputs[p] = {"var":var, "lo":e_lo, "hi":e_hi}
            r+=1

        self.btn_run_sobol = ttk.Button(left, text="▶ RUN SOBOL", command=self.on_run_sobol)
        self.btn_run_sobol.pack(pady=10, fill=tk.X)

        # Right: Plot
        right = ttk.Frame(paned)
        paned.add(right, weight=4)
        self.fig_sob = Figure(figsize=(6,5), dpi=100)
        self.ax_sob = self.fig_sob.add_subplot(111)
        self.cv_sob = FigureCanvasTkAgg(self.fig_sob, master=right)
        self.cv_sob.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Init Sobol defaults (triggered once)
        self.root.after(500, self.update_sobol_defaults)

    def update_sobol_defaults(self):
        # Fill Sobol Low/High with +/- 10% of current inputs
        vals = self.get_vals()
        for p, widgets in self.sobol_inputs.items():
            if p in vals:
                try:
                    v = float(vals[p])
                    widgets['lo'].delete(0, tk.END); widgets['lo'].insert(0, f"{v*0.9:.4g}")
                    widgets['hi'].delete(0, tk.END); widgets['hi'].insert(0, f"{v*1.1:.4g}")
                except: pass

    # ------------------ LOGIC ------------------
    def lock_ui(self, lock=True):
        state = 'disabled' if lock else 'normal'
        self.btn_run_full.config(state=state)
        self.btn_run_sens.config(state=state)
        self.btn_run_sobol.config(state=state)
        if lock: 
            self.progress.start(10)
            self.status_label.config(text="Running...", foreground="orange")
        else: 
            self.progress.stop()
            self.status_label.config(text="Ready", foreground="black")

    def get_vals(self):
        v = {}
        for k,e in self.entries.items(): v[k]=e.get()
        v["core"] = self.core_var.get()
        return v
    
    def write_temp_excel(self, mode):
        vals = self.get_vals()
        rows = [{"Key":k, "Value":v} for k,v in vals.items()]
        
        with pd.ExcelWriter(self.temp_excel) as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Inputs", index=False)
            
            if mode == "sens":
                srows = []
                for p, w in self.sens_inputs.items():
                    if w['var'].get():
                        try:
                            pct = float(w['pct'].get()) / 100.0
                            pts = int(w['pts'].get())
                            srows.append({
                                "name": p, "mode":"percent", "delta_percent": pct, 
                                "n_points": pts, "enable": 1
                            })
                        except: pass
                pd.DataFrame(srows).to_excel(writer, sheet_name="SENS", index=False)
            
            if mode == "sobol":
                urows = []
                for p, w in self.sobol_inputs.items():
                    if w['var'].get():
                        try:
                            l = float(w['lo'].get())
                            h = float(w['hi'].get())
                            urows.append({"name":p, "low":l, "high":h, "enable":1})
                        except: pass
                pd.DataFrame(urows).to_excel(writer, sheet_name="UQ", index=False)
                
                N = self.ent_sobol_n.get()
                pd.DataFrame([{"key":"N_base", "value":N}]).to_excel(writer, sheet_name="UQ_Control", index=False)

    # --- HANDLERS ---
    def on_run_full(self):
        if self.is_running: return
        self.is_running = True
        self.lock_ui(True)
        threading.Thread(target=self.worker_full).start()

    def worker_full(self):
        try:
            self.write_temp_excel("full")
            vals = self.get_vals()
            v_clean = {k: (float(v) if k not in ["face_angles","web_angles","core"] else v) for k,v in vals.items()}
            v_clean["face_angles"]=vals["face_angles"]; v_clean["web_angles"]=vals["web_angles"]
            core = vals["core"]
            PPW = int(v_clean["PPW"])
            nx_min = int(v_clean["nx_min"])
            
            if core=="m2":
                Pcr = eval_m2_Pcr(v_clean, PPW=PPW, nx_min=nx_min)
                res = {"core":"m2", "Pcr": Pcr, "has_mode":False}
            else:
                mode = eval_m3_Pcr_and_mode(v_clean, PPW=PPW, nx_min=nx_min, return_mode=True)
                P, dlin, dloc, dtot, dcr, _ = koiter_curves_from_mode(mode)
                res = {"core":"m3", "Pcr":mode["Pcr"], "dcr": dcr, "has_mode":True, "P":P, "dtot":dtot, "dlin":dlin, "mode_data": mode}
            
            self.root.after(0, self.done_full, res)
        except Exception as e:
            self.root.after(0, self.fail_any, str(e))

    def done_full(self, res):
        self.is_running=False
        self.lock_ui(False)
        txt = f"Core: {res['core']}\nPcr = {res['Pcr']:.2f} N\n"
        if res['has_mode']: txt+=f"δcr = {res['dcr']*1000:.3f} mm\n"
        self.txt_full.delete(1.0, tk.END); self.txt_full.insert(tk.END, txt)
        
        self.ax_ld.clear()
        if res['has_mode']:
            self.ax_ld.plot(res['dtot'], res['P'], label='Total')
            self.ax_ld.plot(res['dlin'], res['P'], '--', alpha=0.5, label='Linear')
            self.ax_ld.scatter([res['dcr']], [res['Pcr']], c='r', zorder=5)
            self.ax_ld.legend(); self.ax_ld.grid(True, alpha=0.3)
            self.ax_ld.set_xlabel("Deflection [m]"); self.ax_ld.set_ylabel("Load [N]")
        self.cv_ld.draw()
        
        self.ax_ct.clear()
        if res['has_mode']:
            md = res['mode_data']
            plot_contour_on_ax(md['x'], md['b'], md['F'], "Mode", self.ax_ct)
        self.cv_ct.draw()

    # --- SENS ---
    def on_run_sens(self):
        if self.is_running: return
        self.is_running=True; self.lock_ui(True)
        threading.Thread(target=self.worker_sens).start()

    def worker_sens(self):
        try:
            self.write_temp_excel("sens")
            out = run_sens(self.temp_excel, make_plot=False)
            self.root.after(0, self.done_sens, out)
        except Exception as e: self.root.after(0, self.fail_any, str(e))

    def done_sens(self, out):
        self.is_running=False; self.lock_ui(False)
        self.fig_sens.clf() # Clear whole figure
        
        df = pd.DataFrame(out['df_results'])
        if df.empty: 
            self.cv_sens.draw()
            return
            
        names = sorted(df['name'].unique())
        n = len(names)
        if n == 0: return
        
        # Grid layout
        cols = 2
        rows = math.ceil(n / cols)
        
        axes = self.fig_sens.subplots(rows, cols)
        # flatten if needed
        if n==1: ax_list = [axes]
        else: ax_list = axes.flatten()
        
        base = out['baseline_Pcr']
        
        for i, nm in enumerate(names):
            ax = ax_list[i]
            sub = df[df['name']==nm].sort_values("value")
            ax.plot(sub['value'], sub['P_cr [N]'], 'o-', markersize=4)
            ax.axhline(base, color='r', linestyle='--', alpha=0.5)
            ax.set_title(nm, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide empty
        for i in range(n, len(ax_list)):
            ax_list[i].axis('off')
            
        self.fig_sens.tight_layout()
        self.cv_sens.draw()

    # --- SOBOL ---
    def on_run_sobol(self):
        if self.is_running: return
        self.is_running=True; self.lock_ui(True)
        threading.Thread(target=self.worker_sobol).start()
    
    def worker_sobol(self):
        try:
            self.write_temp_excel("sobol")
            out = run_sobol(self.temp_excel, make_plot=False)
            self.root.after(0, self.done_sobol, out)
        except Exception as e: self.root.after(0, self.fail_any, str(e))

    def done_sobol(self, out):
        self.is_running=False; self.lock_ui(False)
        self.ax_sob.clear()
        
        names = out['names_sorted']
        S = out['S_sorted']
        ST = out['ST_sorted']
        
        x = np.arange(len(names))
        w = 0.35
        self.ax_sob.bar(x - w/2, S, w, label='Si')
        self.ax_sob.bar(x + w/2, ST, w, label='STi')
        self.ax_sob.set_xticks(x)
        self.ax_sob.set_xticklabels(names, rotation=30, ha='right')
        self.ax_sob.legend()
        self.ax_sob.set_ylabel("Sobol Index")
        self.ax_sob.set_title("Sobol Indices")
        self.cv_sob.draw()

    def fail_any(self, msg):
        self.is_running=False
        self.lock_ui(False)
        messagebox.showerror("Error", msg)

    def load_config(self):
        f = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not f: return
        try:
            with open(f,'r') as fp: d=json.load(fp)
            for k,v in d.items():
                if k in self.entries:
                    self.entries[k].delete(0,tk.END); self.entries[k].insert(0,str(v))
            if "core" in d: self.core_var.set(d["core"])
            self.update_sobol_defaults()
        except Exception as e: messagebox.showerror("Err", str(e))

    def save_config(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not f: return
        try:
            d = self.get_vals()
            with open(f,'w') as fp: json.dump(d, fp, indent=2)
        except Exception as e: messagebox.showerror("Err", str(e))

def main():
    root = tk.Tk()
    app = BucklingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

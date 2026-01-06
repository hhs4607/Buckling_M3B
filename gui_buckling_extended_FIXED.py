#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI for Buckling Analysis M3 - ENHANCED Version
--------------------------------------------------------
Enhancements (following CODE_ANALYSIS_REPORT.md):
1. ✅ M2 now shows full load-deflection curves (extracted from M2a)
2. ✅ Responsive window sizing (adapts to screen)
3. ✅ Better layout with resizable panels
4. ✅ Improved labels and spacing
5. ✅ Tooltips for all parameters (hover for descriptions)
6. ✅ Real-time input validation (red/green borders)
7. ✅ Input sanitization for ply angles
8. ✅ JSON config validation on load
9. ✅ Warning for M3 + large N_base
10. ✅ Units displayed in labels
11. ✅ Progress messages with evaluation counts
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
import re
import os
import sys
import webbrowser

# Try to import PIL for high-quality image scaling
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Image features will show text fallback.")

# Import the analysis functions
from buckling_analysis_M3a import (
    eval_m2_Pcr,
    eval_m2_Pcr_with_mode,
    
    eval_m3_Pcr_and_mode,
    koiter_curves_from_mode,
)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
APP_VERSION = "v2.11"

# ========== TOOLTIP CLASS ==========
class ToolTip:
    """Create tooltips for widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Arial", 9))
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ========== PARAMETER TOOLTIPS ==========
PARAM_TOOLTIPS = {
    "L": "Beam length from root to tip (meters)",
    "b_root": "Beam width at root (clamped end) in meters",
    "b_tip": "Beam width at tip (free end) in meters",
    "h_root": "Core/foam height at root in meters",
    "h_tip": "Core/foam height at tip in meters",
    "w_f": "Width of each vertical web flange in meters",
    "t_face_total": "Total thickness of face laminate (both plies) in meters",
    "face_angles": "Ply angles for face laminate (comma-separated, e.g., 0,45,-45,90)",
    "t_web_total": "Total thickness of web laminate (both plies) in meters",
    "web_angles": "Ply angles for web laminate (comma-separated, e.g., 0,90)",
    "Ef": "Fiber longitudinal elastic modulus in Pascals (Pa)",
    "Em": "Matrix elastic modulus in Pascals (Pa)",
    "Gf": "Fiber shear modulus in Pascals (Pa)",
    "nuf": "Fiber Poisson's ratio (dimensionless, typically 0.2-0.3)",
    "num": "Matrix Poisson's ratio (dimensionless, typically 0.3-0.4)",
    "Vf": "Fiber volume fraction (dimensionless, 0 < Vf < 1, typically 0.5-0.7)",
    "Ktheta_root_per_m": "Rotational spring stiffness at root per unit width (N·m/m)",
    "PPW": "Points per wavelength for spatial discretization (higher = more accurate)",
    "nx_min": "Minimum number of grid points along beam length (higher = more accurate)",
}


class BucklingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Embraer BOX BEAM Buckling (E3B) Analysis Program ({APP_VERSION})")

        # Detect screen size and set window to 85% of screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(int(screen_width * 0.85), 1400)
        window_height = min(int(screen_height * 0.85), 850)

        # Center window on screen
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.root.minsize(1000, 600)  # Minimum usable size

        # Image asset paths (user can replace these)
        self.left_geom_img_path = str(BASE_DIR / "assets" / "geom_def_v2.png")  # geome: ...\assets\geom_def_v2.png
        self.logo_banner_img_path = str(BASE_DIR / "assets" / "logos_v3.png")  # logo: ...\logos_v3.png
        self.user_manual_path = self._resource_path(Path("assets") / "manuals" / "User_Manual.pdf")
        self.theory_manual_path = self._resource_path(Path("assets") / "manuals" / "Theory_Manual.pdf")

        # Storage for results
        self.current_results = None
        self.is_running = False

        # Storage for image references (prevent garbage collection)
        self._img_refs = {}

        # Input validation state
        self.validation_state = {}  # Track which inputs are valid

        # Track colorbar to prevent accumulation
        self.colorbar = None

        # Setup custom styles for validation
        self.setup_validation_styles()

        # Create main layout with notebook (tabs)
        self.create_layout()
        self.load_defaults()

    def _set_image_fit_to_container(self, label, img_path, max_h, side_padding=5):
        if not PIL_AVAILABLE:
            return
        # 컨테이너 실제 폭 기준
        w = max(label.master.winfo_width() - 2 * side_padding, 50)
        photo = self._load_and_scale_image(img_path, w, max_h)
        if photo:
            label.config(image=photo)
            self._img_refs[f"img_{id(label)}"] = photo

    # ========== IMAGE HELPER METHODS ==========
    def _load_and_scale_image(self, path, max_w, max_h):
        """
        Load and scale image maintaining aspect ratio.
        Returns ImageTk.PhotoImage or None if failed.
        """
        if not PIL_AVAILABLE:
            return None

        try:
            img = Image.open(path)
            # Calculate scaling to fit within max_w x max_h
            img_w, img_h = img.size
            scale = min(max_w / img_w, max_h / img_h, 1.0)  # Don't upscale
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img_resized)
        except Exception as e:
            print(f"Warning: Could not load image {path}: {e}")
            return None

    def _resource_path(self, relative_path):
        base_dir = Path(getattr(sys, "_MEIPASS", BASE_DIR))
        return base_dir / Path(relative_path)

    def _open_manual(self, manual_path, title):
        try:
            manual_path = Path(manual_path)
            if not manual_path.exists():
                messagebox.showerror("Manual Missing", f"{title} not found:\n{manual_path}")
                return
            if sys.platform.startswith("win"):
                os.startfile(str(manual_path))
            else:
                webbrowser.open(manual_path.as_uri())
        except Exception as exc:
            messagebox.showerror("Open Manual Failed", f"{title}:\n{exc}")

    def _add_manual_buttons(self, parent, row=1, column=0):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=column, sticky="e", pady=(6, 0))
        user_btn = ttk.Button(
            button_frame,
            text="User Manual",
            command=lambda: self._open_manual(self.user_manual_path, "User Manual"),
            width=14,
        )
        user_btn.pack(side=tk.LEFT, padx=(0, 6))
        theory_btn = ttk.Button(
            button_frame,
            text="Theory Manual",
            command=lambda: self._open_manual(self.theory_manual_path, "Theory Manual"),
            width=14,
        )
        theory_btn.pack(side=tk.LEFT)
        ToolTip(user_btn, "Open User Manual (PDF)")
        ToolTip(theory_btn, "Open Theory Manual (PDF)")
        return button_frame

    def _create_header_block(self, parent, title_main, title_sub, desc_text,
                             logo_max_h=120, wraplength=450):
        """
        FULL 탭과 동일한 규격(Header: 좌=Title/Desc, 우=Logo)을 생성.
        parent의 row=0, col=0에 grid로 배치해서 사용하게끔 구성.
        """
        header_frame = ttk.Frame(parent, relief='solid', borderwidth=1)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header_frame.columnconfigure(0, weight=1)  # Col=0
        header_frame.columnconfigure(1, weight=1)  # Col=1 (균등)

        # ---- Col=0 (위/아래: 1:3) ----
        left = ttk.Frame(header_frame)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)  # 위(Title)
        left.rowconfigure(1, weight=3)
        left.rowconfigure(2, weight=0)  # 아래(Description)

        title_frame = ttk.Frame(left)
        title_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(title_frame, text=title_main,
                  font=('Arial', 12, 'bold'), foreground='#2c3e50').pack(anchor='w')
        ttk.Label(title_frame, text=title_sub,
                  font=('Arial', 10), foreground='#34495e').pack(anchor='w')

        desc_label = ttk.Label(left, text=desc_text,
                               font=('Arial', 9), foreground='#555',
                               wraplength=wraplength, justify='left')
        desc_label.grid(row=1, column=0, sticky="nw", pady=(5, 0))

        spacer_frame = ttk.Frame(left)
        spacer_frame.grid(row=2, column=0, sticky="ew")

        # ---- Col=1 (Logo + Manuals) ----
        right = ttk.Frame(header_frame)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        logo_container = ttk.Frame(right)
        logo_container.grid(row=0, column=0, sticky="nsew")
        logo_container.columnconfigure(0, weight=1)
        logo_container.rowconfigure(0, weight=1)

        logo_label = ttk.Label(logo_container, text="", anchor='center')
        logo_label.grid(row=0, column=0, sticky="nsew")

        # ?????? ?????? + ??????????? ????????
        logo_path = Path(self.logo_banner_img_path)
        if PIL_AVAILABLE and logo_path.exists():
            # ?????? ??????
            photo = self._load_and_scale_image(str(logo_path), 520, logo_max_h)
            if photo:
                logo_label.config(image=photo)
                # ref ?????(???????? ?????????????)
                self._img_refs[f"logo_{id(parent)}"] = photo

            # ???????? ????????????? ??????????? (debounced)
            self._bind_resizable_image_debounced(
                logo_label, str(logo_path), max_h=logo_max_h, side_padding=5, delay=60
            )
        else:
            logo_label.config(text="[LOGO]", foreground="#7f8c8d")

        return header_frame

    def _bind_resizable_image(self, label, img_path, max_h, side_padding=10):
        """
        Bind image to resize with parent frame width changes.
        Updates image dynamically on <Configure> event.
        """
        if not PIL_AVAILABLE:
            return

        def on_resize(event):
            # Get available width from parent frame
            avail_w = event.width - 2 * side_padding
            if avail_w <= 0:
                return

            # Load and scale image
            photo = self._load_and_scale_image(img_path, avail_w, max_h)
            if photo:
                label.config(image=photo)
                # Store reference to prevent garbage collection
                self._img_refs[f"img_{id(label)}"] = photo

        # Bind to parent frame configure event
        label.master.bind('<Configure>', on_resize, add='+')

    def _bind_resizable_image_debounced(self, label, img_path, max_h, side_padding=10, delay=50):
        """
        Bind image to resize with debouncing to reduce resize thrashing.
        """
        if not PIL_AVAILABLE:
            return

        resize_timer = None

        def on_resize(event):
            nonlocal resize_timer
            # Cancel pending resize
            if resize_timer is not None:
                label.after_cancel(resize_timer)

            # Schedule new resize
            def do_resize():
                avail_w = event.width - 2 * side_padding
                if avail_w <= 0:
                    return
                photo = self._load_and_scale_image(img_path, avail_w, max_h)
                if photo:
                    label.config(image=photo)
                    self._img_refs[f"img_{id(label)}"] = photo

            resize_timer = label.after(delay, do_resize)

        label.master.bind('<Configure>', on_resize, add='+')

    def create_layout(self):
        """Create the main GUI layout with tabs"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create each tab's content
        self.full_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.full_frame, text="Buckling (Single Case)")

        self.sens_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.sens_frame, text="Sensitivity (OAT)")

        self.sobol_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.sobol_frame, text="Uncertainty (Sobol UQ)")

        # Create each tab's content
        self.create_full_tab()
        self.create_sens_tab()
        self.create_sobol_tab()

    # ========== TAB 1: FULL ANALYSIS ==========
    def create_full_tab(self):
        """Create FULL analysis tab"""

        # Split into left (inputs) and right (results)
        self.full_frame.columnconfigure(1, weight=3)  # Give more weight to right panel
        self.full_frame.rowconfigure(0, weight=1)

        # Left panel - Inputs (scrollable)
        left_frame = ttk.Frame(self.full_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Use flexible width instead of fixed
        canvas = tk.Canvas(left_frame, width=300, highlightthickness=0)
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
        right_frame = ttk.Frame(self.full_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create input sections
        self.create_input_sections()

        # Create results section
        self.create_results_section(right_frame)

        # Mousewheel
        def _on_mousewheel(event):
            # Windows: delta=120 step, macOS: 작은 값
            delta = event.delta
            step = int(-1 * (delta / 120)) if abs(delta) >= 120 else int(-1 * (delta / 30))
            canvas.yview_scroll(step, "units")

        def _bind_wheel(_):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

    def create_input_sections(self):
        """Create all input parameter sections"""
        self.entries = {}
        row = 0

        # ========== GEOMETRY DEFINITION IMAGE (COMPACT) ==========
        geom_img_frame = ttk.Frame(self.scrollable_frame)
        geom_img_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 4), padx=5)
        row += 1

        if PIL_AVAILABLE and Path(self.left_geom_img_path).exists():
            # Image label
            self.geom_img_label = ttk.Label(geom_img_frame, text="", anchor='center')
            self.geom_img_label.pack(fill=tk.X)

            # Initial render: fit to actual container width (avoid clipping/distortion)
            self.root.after(
                50,
                lambda: self._set_image_fit_to_container(
                    self.geom_img_label, self.left_geom_img_path, 120, side_padding=5
                )
            )

            # Bind for dynamic resizing (debounced to avoid thrashing)
            self._bind_resizable_image_debounced(
                self.geom_img_label, self.left_geom_img_path, 120, side_padding=5, delay=60
            )

            # Caption (smaller font, less padding)
            caption_label = ttk.Label(geom_img_frame, text="Geometry Definition",
                                     font=('Arial', 7, 'italic'), foreground='#666')
            caption_label.pack(pady=(1, 2))
        else:
            # Fallback: text box
            fallback_label = ttk.Label(geom_img_frame,
                                      text="[Geometry Diagram]\nPlace image at: assets\\geom_def_v2.png",
                                      font=('Arial', 8, 'italic'), foreground='#999',
                                      background='#f0f0f0', anchor='center', justify='center',
                                      padding=10)
            fallback_label.pack(fill=tk.X)

        # Compact separator
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(4, 6), padx=5)
        row += 1
        # ========== END GEOMETRY IMAGE ==========

        # GEOMETRY
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

        # FACE LAMINATE
        row = self.create_section_header("FACE LAMINATE", row)
        params = [
            ("t_face_total", "Total Thickness (m)", 0.002),
            ("face_angles", "Ply Angles (deg)", "0,45,-45,90"),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # WEB LAMINATE
        row = self.create_section_header("WEB LAMINATE", row)
        params = [
            ("t_web_total", "Total Thickness (m)", 0.0015),
            ("web_angles", "Ply Angles (deg)", "0,90"),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # MATERIAL PROPERTIES
        row = self.create_section_header("MATERIALS", row)
        params = [
            ("Ef", "Fiber Modulus (Pa)", 230e9),
            ("Em", "Matrix Modulus (Pa)", 3.5e9),
            ("Gf", "Fiber Shear Modulus (Pa)", 90e9),
            ("nuf", "Fiber Poisson Ratio", 0.2),
            ("num", "Matrix Poisson Ratio", 0.35),
            ("Vf", "Fiber Volume Fraction", 0.6),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # BOUNDARY CONDITIONS
        row = self.create_section_header("BOUNDARY", row)
        params = [("Ktheta_root_per_m", "Root Spring (N·m/m)", 1e9)]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ANALYSIS SETTINGS
        row = self.create_section_header("SOLVER", row)
        core_frame = ttk.Frame(self.scrollable_frame)
        core_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=10)
        ttk.Label(core_frame, text="Core:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.core_var = tk.StringVar(value="m3")
        ttk.Radiobutton(core_frame, text="M3 (Accurate)", variable=self.core_var, value="m3").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(core_frame, text="M2 (Fast)", variable=self.core_var, value="m2").pack(side=tk.LEFT, padx=5)
        row += 1

        params = [
            ("PPW", "Points/Wavelength", 60),
            ("nx_min", "Min Grid Points", 1801),
        ]
        for key, label, default in params:
            row = self.create_input_row(key, label, default, row)

        # ACTION BUTTONS
        row += 1
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=15)
        self.run_button = ttk.Button(button_frame, text="▶ RUN", command=self.run_full_analysis, width=18)
        self.run_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📁 Load", command=self.load_config, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Save", command=self.save_config, width=10).pack(side=tk.LEFT, padx=5)

        # Progress
        row += 1
        self.progress = ttk.Progressbar(self.scrollable_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)
        row += 1
        self.status_label = ttk.Label(self.scrollable_frame, text="Ready", foreground="green", font=('Arial', 9, 'bold'))
        self.status_label.grid(row=row, column=0, columnspan=2, pady=5)

    def create_section_header(self, title, row):
        frame = ttk.Frame(self.scrollable_frame)
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(12, 5), padx=5)
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=(0, 5))
        ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack(anchor='w')
        return row + 1

    def create_input_row(self, key, label, default, row):
        label_widget = ttk.Label(self.scrollable_frame, text=label)
        label_widget.grid(row=row, column=0, sticky=tk.W, padx=10, pady=2)

        # Add tooltip if available
        if key in PARAM_TOOLTIPS:
            ToolTip(label_widget, PARAM_TOOLTIPS[key])

        entry = ttk.Entry(self.scrollable_frame, width=20)
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=2)
        entry.insert(0, str(default))
        self.entries[key] = entry

        # Add real-time validation
        entry.bind('<KeyRelease>', lambda e, k=key, ent=entry: self.validate_field(k, ent))
        entry.bind('<FocusOut>',  lambda e, k=key, ent=entry: self.validate_field(k, ent))

        # Initialize validation state
        self.validation_state[key] = True

        return row + 1

    def create_results_section(self, parent):
        """Create results display for FULL tab with 2-column header layout"""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)  # Give weight to plot area

        # ========== Row 0: Header (2 columns × 3 rows) ==========
        header_frame = ttk.Frame(parent, relief='solid', borderwidth=1)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Equal width columns
        header_frame.columnconfigure(0, weight=1)  # Left: title + description
        header_frame.columnconfigure(1, weight=1)  # Right: logo

        # Left column (col=0): 3 rows with different weights
        header_left = ttk.Frame(header_frame)
        header_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        header_left.columnconfigure(0, weight=1)
        header_left.rowconfigure(0, weight=1)  # Title row
        header_left.rowconfigure(1, weight=3)  # Description row (3x taller)
        header_left.rowconfigure(2, weight=0)  # Spacer row

        # Row 0: Title block (2 lines)
        title_frame = ttk.Frame(header_left)
        title_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(title_frame, text="Double Tapered Composite Beam: Buckling Analysis",
                  font=('Arial', 12, 'bold'), foreground='#2c3e50').pack(anchor='w')
        ttk.Label(title_frame, text="Single-Case Evaluation Tool",
                  font=('Arial', 10), foreground='#34495e').pack(anchor='w')

        # Row 1: Description block (longer text with wrapping)
        desc_text = (
            "Rayleigh-Ritz energy minimization for local buckling analysis "
            "of tapered composite beams under tip load. "
            "Uses exponential-trigonometric basis functions with spring-supported "
            "root boundary conditions."
        )
        desc_label = ttk.Label(header_left, text=desc_text,
                               font=('Arial', 9), foreground='#555',
                               wraplength=450, justify='left')
        desc_label.grid(row=1, column=0, sticky="nw", pady=(5, 0))

        # Row 2: Spacer (empty or small note)
        spacer_frame = ttk.Frame(header_left)
        spacer_frame.grid(row=2, column=0, sticky="ew")

        # Right column (col=1): Logo + Manuals
        logo_frame = ttk.Frame(header_frame)
        logo_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=10, pady=8)
        logo_frame.columnconfigure(0, weight=1)
        logo_frame.rowconfigure(0, weight=1)
        logo_frame.rowconfigure(1, weight=0)

        logo_container = ttk.Frame(logo_frame)
        logo_container.grid(row=0, column=0, sticky="nsew")
        logo_container.columnconfigure(0, weight=1)
        logo_container.rowconfigure(0, weight=1)

        # Logo label (centered)
        self.logo_banner_label = ttk.Label(logo_container, text="", anchor='center')
        self.logo_banner_label.grid(row=0, column=0, sticky="nsew")

        # Load logo image
        logo_path = BASE_DIR / "assets" / "logos_v3.png"
        if PIL_AVAILABLE and logo_path.exists():
            # 초기 렌더(실제 폭 기준)
            self.root.after(50, lambda: self._set_image_fit_to_container(self.logo_banner_label,
                                                                         str(logo_path), 120, 5))
            # 리사이즈 반응
            self._bind_resizable_image_debounced(self.logo_banner_label, str(logo_path),
                                                 max_h=120, side_padding=5, delay=60)
        else:
            self.logo_banner_label.config(text="[LOGO]", foreground='#7f8c8d')

        # ========== Row 1: Assumptions ==========
        assumptions_frame = ttk.LabelFrame(parent, text="Full Analysis: Description", padding="8")
        assumptions_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        assumptions_text = (
            "• Method: Rayleigh–Ritz energy minimization with exponential-trigonometric basis\n"
            "• Kinematics: 1 mode in x-direction, 1 mode in y-direction (local buckling)\n"
            "• Boundary: Root spring support (Kθ), free tip with axial load\n"
            "• Outputs: Pcr, δcr, mode shape (α*, β*, λx*), load–deflection curves\n"
            "• Solver: M3 (two-term, accurate) / M2 (one-term, fast approximation)"
        )
        ttk.Label(assumptions_frame, text=assumptions_text,
                  font=('Arial', 8), foreground='#555', justify='left').pack(anchor='w')

        # ========== Row 2: Summary ==========
        text_frame = ttk.LabelFrame(parent, text="Summary", padding="10")
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        text_inner = ttk.Frame(text_frame)
        text_inner.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_inner.rowconfigure(0, weight=1)
        text_inner.columnconfigure(0, weight=1)

        self.results_text = tk.Text(text_inner, height=10, font=('Courier', 9), wrap=tk.WORD)
        ys = ttk.Scrollbar(text_inner, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=ys.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        ys.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # ========== Row 3: Plots ==========
        plot_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        plot_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.ax1.set_xlabel("Tip Deflection δ [m]")
        self.ax1.set_ylabel("Load P [N]")
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.ax1.text(0.5, 0.5, 'Run analysis to display results',
                      ha='center', va='center', transform=self.ax1.transAxes,
                      fontsize=11, alpha=0.5, style='italic')

        self.ax2.set_xlabel("x [m]")
        self.ax2.set_ylabel("y [m]")
        self.ax2.text(0.5, 0.5, 'Mode contour will appear here',
                      ha='center', va='center', transform=self.ax2.transAxes,
                      fontsize=11, alpha=0.5, style='italic')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ========== Row 4: Bottom buttons (3 zones) ==========
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=4, column=0, sticky="ew", pady=8)
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=0)
        bottom_frame.columnconfigure(2, weight=1)
        bottom_frame.columnconfigure(3, weight=0)

        export_frame = ttk.Frame(bottom_frame)
        export_frame.grid(row=0, column=1)
        ttk.Button(export_frame, text="📊 Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="📄 Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)

        manual_frame = ttk.Frame(bottom_frame)
        manual_frame.grid(row=0, column=3, sticky="e")
        self._add_manual_buttons(manual_frame, row=0, column=0)

    # ========== TAB 2: SENS ANALYSIS (Simplified) ==========
    def create_sens_tab(self):
        """Create sensitivity analysis tab (param list on top, controls at bottom)"""
         # ---- content: left / right using grid (no pack inside this parent) ----
        content = ttk.Frame(self.sens_frame)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        content.columnconfigure(0, weight=0)  # left
        content.columnconfigure(1, weight=1)  # right grows
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(content, padding="5")
        right.grid(row=0, column=1, sticky="nsew")

        # left: top(list) / bottom(controls)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)  # list grows
        left.rowconfigure(1, weight=0)  # controls fixed-ish

        # ===================== TOP: parameter list (scrollable) =====================
        list_frame = ttk.Frame(left)
        list_frame.grid(row=0, column=0, sticky="nsew")

        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)

        ttk.Label(list_frame, text="Select parameters:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        canvas_sens = tk.Canvas(list_frame, highlightthickness=0)
        scroll_sens = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_sens.yview)
        canvas_sens.grid(row=1, column=0, sticky="nsew")
        scroll_sens.grid(row=1, column=1, sticky="ns")

        param_frame = ttk.Frame(canvas_sens)
        param_frame.bind("<Configure>", lambda e: canvas_sens.configure(scrollregion=canvas_sens.bbox("all")))
        canvas_sens.create_window((0, 0), window=param_frame, anchor="nw")
        canvas_sens.configure(yscrollcommand=scroll_sens.set)

        # ===================== build param rows =====================
        self.sens_vars = {}
        sens_params = [
            ("L", "Beam Length (m)"),
            ("b_root", "Root Width (m)"),
            ("b_tip", "Tip Width (m)"),
            ("h_root", "Root Core Height (m)"),
            ("h_tip", "Tip Core Height (m)"),
            ("w_f", "Flange Width (m)"),
            ("t_face_total", "Face Thickness (m)"),
            ("t_web_total", "Web Thickness (m)"),
            ("Ef", "Fiber Modulus (Pa)"),
            ("Em", "Matrix Modulus (Pa)"),
            ("Gf", "Fiber Shear Modulus (Pa)"),
            ("nuf", "Fiber Poisson Ratio"),
            ("num", "Matrix Poisson Ratio"),
            ("Vf", "Fiber Volume Fraction"),
            ("Ktheta_root_per_m", "Root Spring (N·m/m)"),
            ("PPW", "Points/Wavelength"),
            ("nx_min", "Min Grid Points"),
        ]

        for param_key, param_label in sens_params:
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2, padx=5)

            var = tk.BooleanVar(value=False)
            check = ttk.Checkbutton(frame, text=param_label, variable=var, width=24)
            check.pack(side=tk.LEFT)

            ttk.Label(frame, text="±%:").pack(side=tk.LEFT, padx=(5, 2))
            pct_entry = ttk.Entry(frame, width=8)
            pct_entry.insert(0, "10")
            pct_entry.pack(side=tk.LEFT, padx=2)

            ttk.Label(frame, text="Pts:").pack(side=tk.LEFT, padx=(5, 2))
            pts_entry = ttk.Entry(frame, width=6)
            pts_entry.insert(0, "5")
            pts_entry.pack(side=tk.LEFT)

            self.sens_vars[param_key] = {"enabled": var, "pct": pct_entry, "pts": pts_entry}

        # ===================== BOTTOM: controls (solver/run/progress) =====================
        controls_frame = ttk.Frame(left)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls_frame.columnconfigure(0, weight=1)

        solver_frame = ttk.LabelFrame(controls_frame, text="Solver", padding="10")
        solver_frame.grid(row=0, column=0, sticky="ew")
        self.sens_core_var = tk.StringVar(value="m2")
        ttk.Radiobutton(solver_frame, text="M3", variable=self.sens_core_var, value="m3").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(solver_frame, text="M2", variable=self.sens_core_var, value="m2").pack(side=tk.LEFT, padx=10)

        ttk.Button(controls_frame, text="▶ RUN SENSITIVITY", command=self.run_sens_analysis, width=22).grid(
            row=1, column=0, pady=(8, 6), sticky="ew"
        )

        self.sens_progress = ttk.Progressbar(controls_frame, mode='determinate')
        self.sens_progress.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.sens_status = ttk.Label(controls_frame, text="Ready", foreground="green", font=('Arial', 9, 'bold'))
        self.sens_status.grid(row=3, column=0, sticky="w")

        # ===================== RIGHT: header + desc + plot =====================
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)  # plot row expand

        desc_text = (
            "One-at-a-time (OAT) sensitivity analysis for critical buckling load.\n"
            "Each selected parameter is perturbed around the baseline (±%) and Pcr is evaluated.\n"
            "Outputs: Pcr trends per parameter, baseline reference, exportable figure."
        )

        self._create_header_block(
            right,
            title_main="Double Tapered Composite Beam: OAT Sensitivity Analysis",
            title_sub="One-at-a-time Parameter Sweep",
            desc_text=desc_text,
        )

        assumptions_frame = ttk.LabelFrame(right, text="Sensitivity Analysis: Description", padding="8")
        assumptions_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        assumptions_text = (
            "• Method: One-at-a-time parameter sweep around baseline\n"
            "• Sweep: linear grid within ±% range with specified points\n"
            "• Output: Pcr vs parameter value, baseline shown as dashed line\n"
            "• Solver: M2 (fast) / M3 (accurate)\n"
        )
        ttk.Label(assumptions_frame, text=assumptions_text,
                  font=('Arial', 8), foreground='#555', justify='left').pack(anchor='w')

        plot_frame = ttk.LabelFrame(right, text="Sensitivity Results", padding="10")
        plot_frame.grid(row=2, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.sens_fig = Figure(figsize=(7, 6), dpi=100)
        self.sens_canvas = FigureCanvasTkAgg(self.sens_fig, master=plot_frame)
        self.sens_canvas.draw()
        self.sens_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        export_frame = ttk.Frame(right)
        export_frame.grid(row=3, column=0, sticky="e", pady=(8, 0))
        ttk.Button(export_frame, text="📊 Export", command=self.export_sens_plot).pack(side=tk.RIGHT)

    # ========== TAB 3: SOBOL ANALYSIS (Simplified) ==========
    def create_sobol_tab(self):
        """Create Sobol/UQ analysis tab (param list on top, controls at bottom)"""
        content = ttk.Frame(self.sobol_frame)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(content, padding="5")
        right.grid(row=0, column=1, sticky="nsew")

        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)  # list grows
        left.rowconfigure(1, weight=0)  # controls bottom

        # ===================== TOP: uncertain parameters list =====================
        list_frame = ttk.Frame(left)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)

        ttk.Label(list_frame, text="Uncertain Parameters:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        canvas_sobol = tk.Canvas(list_frame, highlightthickness=0)
        scroll_sobol = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_sobol.yview)
        canvas_sobol.grid(row=1, column=0, sticky="nsew")
        scroll_sobol.grid(row=1, column=1, sticky="ns")

        uq_frame = ttk.Frame(canvas_sobol)
        uq_frame.bind("<Configure>", lambda e: canvas_sobol.configure(scrollregion=canvas_sobol.bbox("all")))
        canvas_sobol.create_window((0, 0), window=uq_frame, anchor="nw")
        canvas_sobol.configure(yscrollcommand=scroll_sobol.set)

        self.sobol_vars = {}

        sobol_defaults = [
            ("L", "Beam Length (m)", 1.35, 1.65),
            ("b_root", "Root Width (m)", 0.072, 0.088),
            ("b_tip", "Tip Width (m)", 0.036, 0.044),
            ("h_root", "Root Core Height (m)", 0.0225, 0.0275),
            ("h_tip", "Tip Core Height (m)", 0.0135, 0.0165),
            ("w_f", "Flange Width (m)", 0.018, 0.022),
            ("t_face_total", "Face Thickness (m)", 0.0018, 0.0022),
            ("t_web_total", "Web Thickness (m)", 0.00135, 0.00165),
            ("Ef", "Fiber Modulus (Pa)", 207e9, 253e9),
            ("Em", "Matrix Modulus (Pa)", 3.15e9, 3.85e9),
            ("Gf", "Fiber Shear Modulus (Pa)", 81e9, 99e9),
            ("nuf", "Fiber Poisson Ratio", 0.18, 0.22),
            ("num", "Matrix Poisson Ratio", 0.315, 0.385),
            ("Vf", "Fiber Volume Fraction", 0.54, 0.66),
            ("Ktheta_root_per_m", "Root Spring (N·m/m)", 0.9e9, 1.1e9),
            ("PPW", "Points/Wavelength", 54, 66),
            ("nx_min", "Min Grid Points", 1621, 1981),
        ]

        for param_key, param_label, low_default, high_default in sobol_defaults:
            frame = ttk.Frame(uq_frame)
            frame.pack(fill=tk.X, pady=3, padx=5)

            var = tk.BooleanVar(value=False)
            check = ttk.Checkbutton(frame, text=param_label, variable=var, width=22)
            check.pack(side=tk.LEFT)

            ttk.Label(frame, text="Low:").pack(side=tk.LEFT, padx=(5, 2))
            low_entry = ttk.Entry(frame, width=10)
            low_entry.insert(0, f"{low_default:.4g}")
            low_entry.pack(side=tk.LEFT, padx=2)

            ttk.Label(frame, text="High:").pack(side=tk.LEFT, padx=(5, 2))
            high_entry = ttk.Entry(frame, width=10)
            high_entry.insert(0, f"{high_default:.4g}")
            high_entry.pack(side=tk.LEFT)

            self.sobol_vars[param_key] = {"enabled": var, "low": low_entry, "high": high_entry}

        # ===================== BOTTOM: controls =====================
        controls_frame = ttk.Frame(left)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls_frame.columnconfigure(0, weight=1)

        control_frame = ttk.LabelFrame(controls_frame, text="Settings", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(control_frame, text="N_base:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.sobol_N = ttk.Entry(control_frame, width=12)
        self.sobol_N.insert(0, "100")
        self.sobol_N.grid(row=0, column=1, pady=3, padx=5, sticky="w")

        ttk.Label(control_frame, text="Seed:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.sobol_seed = ttk.Entry(control_frame, width=12)
        self.sobol_seed.insert(0, "1234")
        self.sobol_seed.grid(row=1, column=1, pady=3, padx=5, sticky="w")

        ttk.Label(control_frame, text="Uncertainty %:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.sobol_pct = ttk.Entry(control_frame, width=12)
        self.sobol_pct.insert(0, "10")
        self.sobol_pct.grid(row=2, column=1, pady=3, padx=5, sticky="w")

        ttk.Button(control_frame, text="📐 Update from Baseline",
                   command=self.update_sobol_from_baseline, width=22).grid(
            row=3, column=0, columnspan=2, pady=5, sticky="ew"
        )

        solver_frame = ttk.LabelFrame(controls_frame, text="Solver", padding="10")
        solver_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.sobol_core_var = tk.StringVar(value="m2")
        ttk.Radiobutton(solver_frame, text="M3", variable=self.sobol_core_var, value="m3").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(solver_frame, text="M2 ⭐", variable=self.sobol_core_var, value="m2").pack(side=tk.LEFT, padx=10)

        ttk.Button(controls_frame, text="▶ RUN SOBOL", command=self.run_sobol_analysis, width=22).grid(
            row=2, column=0, pady=(8, 6), sticky="ew"
        )

        self.sobol_progress = ttk.Progressbar(controls_frame, mode='determinate')
        self.sobol_progress.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        self.sobol_status = ttk.Label(controls_frame, text="Ready", foreground="green", font=('Arial', 9, 'bold'))
        self.sobol_status.grid(row=4, column=0, sticky="w")

        # ===================== RIGHT: header + desc + plot =====================
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        desc_text = (
            "Sobol variance-based sensitivity analysis for uncertainty quantification.\n"
            "Computes first-order (S_i) and total-order (S_Ti) indices from random sampling.\n"
            "Outputs: ranked Sobol indices bar chart and exportable figure."
        )

        self._create_header_block(
            right,
            title_main="Double Tapered Composite Beam: Sobol Uncertainty Quantification",
            title_sub="Variance-based Global Sensitivity (S₁, Sᵀ)",
            desc_text=desc_text,
        )

        assumptions_frame = ttk.LabelFrame(right, text="Sobol UQ: Description", padding="8")
        assumptions_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        assumptions_text = (
            "• Method: Sobol indices using two base sample matrices (A, B)\n"
            "• Evaluations: N_base × (2 + k)\n"
            "• Output: S_i (first-order), S_Ti (total-order), sorted by S_Ti\n"
            "• Solver: M2 (fast) / M3 (accurate)\n"
        )
        ttk.Label(assumptions_frame, text=assumptions_text,
                  font=('Arial', 8), foreground='#555', justify='left').pack(anchor='w')

        plot_frame = ttk.LabelFrame(right, text="Sobol Indices", padding="10")
        plot_frame.grid(row=2, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.sobol_fig = Figure(figsize=(7, 6), dpi=100)
        self.sobol_canvas = FigureCanvasTkAgg(self.sobol_fig, master=plot_frame)
        self.sobol_canvas.draw()
        self.sobol_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        export_frame = ttk.Frame(right)
        export_frame.grid(row=3, column=0, sticky="e", pady=(8, 0))
        ttk.Button(export_frame, text="📊 Export", command=self.export_sobol_plot).pack(side=tk.RIGHT)

    # ========== HELPER FUNCTIONS ==========
    def get_values(self):
        """Extract values from entries"""
        vals = {}
        numeric_keys = ["L", "b_root", "b_tip", "h_root", "h_tip", "w_f",
                       "t_face_total", "t_web_total", "Ef", "Em", "Gf",
                       "nuf", "num", "Vf", "Ktheta_root_per_m", "PPW", "nx_min"]
        try:
            for key in numeric_keys:
                vals[key] = float(self.entries[key].get().strip())
            vals["face_angles"] = self.entries["face_angles"].get().strip()
            vals["web_angles"] = self.entries["web_angles"].get().strip()
            return vals
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric value: {e}")
            return None

    def validate_inputs(self, vals):
        """Validate inputs"""
        errors = []
        if vals["L"] <= 0: errors.append("Length must be > 0")
        if vals["b_root"] <= 0: errors.append("Root width must be > 0")
        if not (0 < vals["Vf"] < 1): errors.append("Vf must be 0 < Vf < 1")
        if vals["Ef"] <= 0: errors.append("Ef must be > 0")
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False
        return True

    def validate_field(self, key, entry):
        """Real-time field validation with visual feedback"""
        value = entry.get().strip()

        # Skip empty during typing
        if not value:
            entry.config(style="TEntry")
            self.validation_state[key] = False
            return

        is_valid = False

        try:
            # Ply angles validation (comma-separated numbers)
            if key in ["face_angles", "web_angles"]:
                if self.validate_ply_angles(value):
                    is_valid = True
            else:
                # Numeric validation
                num_val = float(value)

                # Range checks
                if key == "Vf":
                    is_valid = (0 < num_val < 1)
                elif key in ["L", "b_root", "b_tip", "h_root", "h_tip", "w_f",
                           "t_face_total", "t_web_total", "Ef", "Em", "Gf"]:
                    is_valid = (num_val > 0)
                elif key in ["nuf", "num"]:
                    is_valid = (-1 < num_val < 0.5)  # Physical range for Poisson's ratio
                elif key in ["PPW", "nx_min"]:
                    is_valid = (num_val >= 10)
                elif key == "Ktheta_root_per_m":
                    is_valid = (num_val >= 0)
                else:
                    is_valid = True

        except ValueError:
            is_valid = False

        # Apply visual feedback
        if is_valid:
            entry.config(style="Valid.TEntry")
            self.validation_state[key] = True
        else:
            entry.config(style="Invalid.TEntry")
            self.validation_state[key] = False

    def validate_ply_angles(self, angle_str):
        """Validate ply angle string (comma-separated numbers)"""
        try:
            # Remove spaces and split by comma
            parts = [p.strip() for p in angle_str.split(',')]
            if not parts:
                return False

            # Check each part is a valid number
            for part in parts:
                if not part:
                    return False
                float(part)  # Will raise ValueError if invalid

            return True
        except (ValueError, AttributeError):
            return False

    def setup_validation_styles(self):
        """Setup custom styles for validation feedback"""
        style = ttk.Style()

        # Valid input style (green border effect)
        style.configure("Valid.TEntry", fieldbackground="#e8f5e9")

        # Invalid input style (red border effect)
        style.configure("Invalid.TEntry", fieldbackground="#ffebee")

    # ========== FULL ANALYSIS ==========
    def run_full_analysis(self):
        """Run FULL mode analysis"""
        if self.is_running:
            return
        vals = self.get_values()
        if vals is None or not self.validate_inputs(vals):
            return

        self.is_running = True
        self.run_button.config(state='disabled')
        self.progress.start(10)
        self.status_label.config(text="Running...", foreground="orange")

        thread = threading.Thread(target=self.full_worker, args=(vals,))
        thread.start()

    def full_worker(self, vals):
        """Worker for FULL analysis"""
        try:
            core = self.core_var.get()
            PPW = int(vals["PPW"])
            nx_min = int(vals["nx_min"])

            if core == "m2":
                # ✅ NEW: Use enhanced M2 with Koiter curves
                mode = eval_m2_Pcr_with_mode(vals, PPW=PPW, nx_min=nx_min, return_mode=True)
                results = {
                    "core": "M2",
                    "Pcr": mode["Pcr"],
                    "dcr": mode["dcr"],
                    "alpha_star": mode["alpha_star"],
                    "beta_star": mode["beta_star"],
                    "lambda_x": mode["lambda_x"],
                    "mode_available": True,  # ✅ Enable plotting!
                    "P": mode["P"],
                    "dlin": mode["dlin"],
                    "dloc": mode["dloc"],
                    "dtot": mode["dtot"],
                    "mode": mode
                }
            else:
                mode = eval_m3_Pcr_and_mode(vals, PPW=PPW, nx_min=nx_min, return_mode=True)
                P, dlin, dloc, dtot, dcr, more = koiter_curves_from_mode(mode)
                results = {
                    "core": "M3", "Pcr": mode["Pcr"], "dcr": dcr,
                    "alpha_star": mode["alpha_star"], "beta_star": mode["beta_star"],
                    "lambda_x": mode["lambda_x"], "mode_available": True,
                    "P": P, "dlin": dlin, "dloc": dloc, "dtot": dtot, "mode": mode
                }
            self.root.after(0, self.full_complete, results)
        except Exception as e:
            self.root.after(0, self.full_error, str(e))

    def full_complete(self, results):
        """FULL analysis complete"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Complete ✓", foreground="green")
        self.current_results = results
        self.display_full_results(results)

        # ✅ FIX: Always plot results (both M2 and M3)
        self.plot_full_results(results)

    def full_error(self, msg):
        """FULL analysis error"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Failed ✗", foreground="red")
        messagebox.showerror("Error", f"Analysis failed:\n{msg}")

    def display_full_results(self, results):
        """Display FULL results"""
        self.results_text.delete(1.0, tk.END)
        text = f"{'-'*50}\n"
        text += f"Solver: {results['core']}\n{'-'*50}\n\n"
        text += f"Critical Load:\n  Pcr = {results['Pcr']:.2f} N\n"
        text += f"      = {results['Pcr']/1000:.3f} kN\n\n"
        if results["mode_available"]:
            text += f"Critical Deflection:\n  δcr = {results['dcr']*1000:.4f} mm\n\n"
            text += f"Mode Parameters:\n"
            text += f"  α* = {results['alpha_star']:.4f} 1/m\n"
            text += f"  β* = {results['beta_star']:.4f} 1/m\n"
            text += f"  λx* = {results['lambda_x']:.4f} m\n"
        else:
            text += f"\nNote: M2 provides Pcr only.\n"
            text += f"Use M3 for full mode shapes\nand load-deflection curves.\n"
        self.results_text.insert(1.0, text)

    def plot_full_results(self, results):
        """Plot FULL results - 2 figures side by side"""
        self.ax1.clear()
        self.ax2.clear()

        # Clear old colorbar if it exists
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except (KeyError, ValueError, AttributeError, Exception):
                pass  # Already removed or doesn't exist
            self.colorbar = None

        if results["mode_available"]:
            # LEFT PLOT: Load-deflection curves
            P = results["P"]; dlin = results["dlin"]; dloc = results["dloc"]; dtot = results["dtot"]
            Pcr = results["Pcr"]; dcr = results["dcr"]
            dmax = 1.5 * dcr; Pmax = 1.5 * Pcr

            def clip(P_arr, d_arr):
                m = (P_arr <= Pmax) & (d_arr <= dmax + 1e-15)
                return d_arr[m], P_arr[m]

            dloc_c, P_loc_c = clip(P, dloc)
            dtot_c, P_tot_c = clip(P, dtot)
            fb = float(np.dot(P, dlin) / max(np.dot(P, P), 1e-18))
            d_lin = np.linspace(0.0, dmax, 400)
            P_lin = d_lin / max(fb, 1e-18)

            self.ax1.plot(d_lin, P_lin, '--', lw=2, label='Linear', color='blue')
            self.ax1.plot(dloc_c, P_loc_c, ':', lw=2.2, alpha=0.95, label='Nonlinear', color='orange')
            self.ax1.plot(dtot_c, P_tot_c, '-', lw=2.4, label='Total', color='green')

            d_at_Pcr = float(np.interp(Pcr, P_tot_c if len(P_tot_c) > 2 else P, dtot_c if len(dtot_c) > 2 else dtot))
            self.ax1.scatter([d_at_Pcr], [Pcr], s=70, ec='k', zorder=4, color='red', label=f'Pcr={Pcr:.0f}N')

            self.ax1.set_xlabel("Tip Deflection δ [m]"); self.ax1.set_ylabel("Load P [N]")
            self.ax1.set_xlim(0, dmax); self.ax1.set_ylim(0, Pmax)
            self.ax1.grid(True, ls='--', alpha=0.3); self.ax1.legend(loc='lower right')
            solver_name = results.get("core", "M3")
            self.ax1.set_title(f"Load-Deflection Curve ({solver_name})")

            # RIGHT PLOT: Mode Contour
            self.plot_mode_contour(results, self.ax2)

        else:
            # ✅ M2: Show bar chart with Pcr value on left, message on right
            Pcr = results["Pcr"]

            self.ax1.bar([0], [Pcr], width=0.5, color='steelblue', edgecolor='black', linewidth=2)
            self.ax1.set_xlim(-0.5, 0.5)
            self.ax1.set_ylim(0, Pcr * 1.2)
            self.ax1.set_xticks([0])
            self.ax1.set_xticklabels(['M2 Solver'])
            self.ax1.set_ylabel("Critical Buckling Load Pcr [N]", fontsize=11)
            self.ax1.set_title("Critical Load (M2 Fast Approximation)", fontsize=12, fontweight='bold')
            self.ax1.grid(True, axis='y', ls='--', alpha=0.3)

            # Add value label on bar
            self.ax1.text(0, Pcr, f'  {Pcr:.1f} N\n  ({Pcr/1000:.2f} kN)',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Right plot: informational message
            self.ax2.text(0.5, 0.5, 'M2 provides Pcr only.\nUse M3 for full mode shapes\nand contour plots.',
                        transform=self.ax2.transAxes, ha='center', va='center',
                        fontsize=11, style='italic', color='gray',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            self.ax2.set_xticks([]); self.ax2.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_mode_contour(self, results, ax):
        """Plot mode shape contour on given axis"""
        mode = results.get("mode", {})
        x = mode.get("x", np.array([]))
        b = mode.get("b", np.array([]))
        F = mode.get("F", np.array([]))
        ky = mode.get("ky", np.array([]))

        if len(x) == 0 or len(F) == 0:
            ax.text(0.5, 0.5, 'Mode data not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, alpha=0.5, style='italic')
            return

        # Build contour grid (reduced resolution for GUI)
        nx_grid, ny_grid = 80, 40
        X_lin = np.linspace(x[0], x[-1], nx_grid)
        bX = np.interp(X_lin, x, b)

        # Create 2D grids
        X2d = np.zeros((nx_grid, ny_grid))
        Y2d = np.zeros((nx_grid, ny_grid))
        W = np.zeros((nx_grid, ny_grid))

        FX = np.interp(X_lin, x, F)

        for i in range(nx_grid):
            y_phys = np.linspace(0.0, bX[i], ny_grid)
            y_plot = y_phys - bX[i]/2.0  # Center around y=0

            X2d[i, :] = X_lin[i]
            Y2d[i, :] = y_plot

            ky_i = np.interp(X_lin[i], x, ky)
            W[i, :] = float(FX[i]) * np.sin(ky_i * y_phys)

        # Normalize for plotting
        maxabs = max(np.max(np.abs(W)), 1e-18)
        Wn = W / maxabs

        # Plot contour
        pcm = ax.pcolormesh(X2d, Y2d, Wn, cmap='viridis', shading='gouraud')
        ax.set_aspect('auto')
        ax.set_xlabel('x [m]', fontsize=9)
        ax.set_ylabel('y [m]', fontsize=9)
        solver_name = results.get("core", "M3")
        ax.set_title(f"Mode Shape Contour ({solver_name})", fontsize=10)

        # Add new colorbar (old one already removed in plot_full_results)
        self.colorbar = self.fig.colorbar(pcm, ax=ax, label='w/max|w|', fraction=0.046, pad=0.04)

    # ========== SENS ANALYSIS (Same as before) ==========
    def run_sens_analysis(self):
        """Run sensitivity analysis"""
        if self.is_running:
            return

        vals = self.get_values()
        if vals is None or not self.validate_inputs(vals):
            return

        enabled = [(k, v) for k, v in self.sens_vars.items() if v["enabled"].get()]
        if not enabled:
            messagebox.showwarning("Warning", "Select at least one parameter")
            return

        self.is_running = True
        self.sens_status.config(text="Running...", foreground="orange")

        thread = threading.Thread(target=self.sens_worker, args=(vals, enabled))
        thread.start()

    def sens_worker(self, vals0, enabled):
        """SENS worker"""
        try:
            core = self.sens_core_var.get()
            results = []
            total = sum(int(v["pts"].get()) for _, v in enabled)
            count = 0

            for param, widgets in enabled:
                pct = float(widgets["pct"].get()) / 100.0
                n = int(widgets["pts"].get())
                v0 = float(vals0[param])
                grid = np.linspace(v0 * (1 - pct), v0 * (1 + pct), n)

                for val in grid:
                    v = vals0.copy()
                    v[param] = float(val)

                    if core == "m3":
                        Pcr = eval_m3_Pcr_and_mode(v, return_mode=False)
                    else:
                        Pcr = eval_m2_Pcr(v)

                    results.append({"param": param, "value": val, "Pcr": Pcr})
                    count += 1
                    progress = int(100 * count / total)
                    # Update progress bar and status message
                    self.root.after(0, lambda p=progress, c=count, t=total: (
                        self.sens_progress.config(value=p),
                        self.sens_status.config(text=f"Evaluating case {c}/{t}...", foreground="orange")
                    ))

            self.root.after(0, self.sens_complete, results, vals0, core)
        except Exception as e:
            self.root.after(0, self.sens_error, str(e))

    def sens_complete(self, results, vals0, core):
        """SENS complete"""
        self.is_running = False
        self.sens_status.config(text="Complete ✓", foreground="green")
        self.sens_progress.config(value=100)

        self.sens_fig.clear()
        df = pd.DataFrame(results)
        params = df["param"].unique()

        n = len(params)
        cols = 2
        rows = (n + cols - 1) // cols

        for i, param in enumerate(params):
            ax = self.sens_fig.add_subplot(rows, cols, i + 1)
            data = df[df["param"] == param]
            ax.plot(data["value"], data["Pcr"], 'o-', lw=2, markersize=6)
            ax.set_xlabel(param, fontsize=9); ax.set_ylabel("Pcr [N]", fontsize=9)
            ax.grid(True, ls='--', alpha=0.3)
            ax.axvline(vals0[param], color='r', ls='--', alpha=0.5, lw=1.5, label='Baseline')
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)

        self.sens_fig.suptitle(f"Sensitivity Analysis (core={core})", fontsize=11, fontweight='bold')
        self.sens_fig.tight_layout()
        self.sens_canvas.draw()

        self.sens_results = results

    def sens_error(self, msg):
        """SENS error"""
        self.is_running = False
        self.sens_status.config(text="Failed ✗", foreground="red")
        messagebox.showerror("Error", f"SENS failed:\n{msg}")

    # ========== SOBOL ANALYSIS (Same as before) ==========
    def run_sobol_analysis(self):
        """Run Sobol analysis"""
        if self.is_running:
            return

        vals = self.get_values()
        if vals is None or not self.validate_inputs(vals):
            return

        enabled = [(k, v) for k, v in self.sobol_vars.items() if v["enabled"].get()]
        if not enabled:
            messagebox.showwarning("Warning", "Select at least one UQ parameter")
            return

        try:
            uq_params = {}
            for param, widgets in enabled:
                low = float(widgets["low"].get())
                high = float(widgets["high"].get())
                if low >= high:
                    raise ValueError(f"{param}: low must be < high")
                uq_params[param] = {"low": low, "high": high}
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        try:
            N_base = int(self.sobol_N.get())
            seed = int(self.sobol_seed.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid N_base or seed")
            return

        # ✅ WARNING: Check for M3 + large N_base combination
        core = self.sobol_core_var.get()
        k = len(uq_params)
        total_evals = N_base * (2 + k)
        if core == "m3" and N_base > 50:
            est_time_min = int(total_evals * 10 / 60)  # Assume 10 sec per M3 eval
            response = messagebox.askyesno(
                "Performance Warning",
                f"M3 solver with N={N_base} will require ~{total_evals} evaluations.\n"
                f"Estimated time: {est_time_min} minutes or more.\n\n"
                f"Recommendations:\n"
                f"• Use M2 solver (much faster, ~90% accurate)\n"
                f"• Reduce N_base to 30-50\n\n"
                f"Continue anyway?"
            )
            if not response:
                return

        self.is_running = True
        self.sobol_status.config(text="Running...", foreground="orange")

        thread = threading.Thread(target=self.sobol_worker, args=(vals, uq_params, N_base, seed))
        thread.start()

    def sobol_worker(self, vals0, uq_params, N_base, seed):
        """Sobol worker"""
        try:
            core = self.sobol_core_var.get()
            names = list(uq_params.keys())
            k = len(names)

            rng = np.random.default_rng(seed)

            def sample(low, high, N):
                return low + (high - low) * rng.random(N)

            A = np.vstack([sample(uq_params[n]["low"], uq_params[n]["high"], N_base) for n in names]).T
            B = np.vstack([sample(uq_params[n]["low"], uq_params[n]["high"], N_base) for n in names]).T

            total_evals = N_base * 2 + N_base * k
            count = 0

            def eval_row(row):
                v = vals0.copy()
                for j, name in enumerate(names):
                    v[name] = float(row[j])
                if core == "m3":
                    return float(eval_m3_Pcr_and_mode(v, return_mode=False))
                else:
                    return float(eval_m2_Pcr(v))

            YA = []
            for n in range(N_base):
                YA.append(eval_row(A[n, :]))
                count += 1
                # Update progress with evaluation count
                self.root.after(0, lambda c=count, t=total_evals: (
                    self.sobol_progress.config(value=int(100*c/t)),
                    self.sobol_status.config(text=f"Evaluating case {c}/{t}...", foreground="orange")
                ))
            YA = np.array(YA)

            YB = []
            for n in range(N_base):
                YB.append(eval_row(B[n, :]))
                count += 1
                self.root.after(0, lambda c=count, t=total_evals: (
                    self.sobol_progress.config(value=int(100*c/t)),
                    self.sobol_status.config(text=f"Evaluating case {c}/{t}...", foreground="orange")
                ))
            YB = np.array(YB)

            YAB_list = []
            for i in range(k):
                M = A.copy()
                M[:, i] = B[:, i]
                YAB = []
                for n in range(N_base):
                    YAB.append(eval_row(M[n, :]))
                    count += 1
                    self.root.after(0, lambda c=count, t=total_evals: (
                        self.sobol_progress.config(value=int(100*c/t)),
                        self.sobol_status.config(text=f"Evaluating case {c}/{t}...", foreground="orange")
                    ))
                YAB_list.append(np.array(YAB))

            Y_all = np.concatenate([YA, YB])
            V = np.var(Y_all, ddof=1) if len(Y_all) > 1 else 1.0

            S = np.zeros(k)
            ST = np.zeros(k)
            for i in range(k):
                YAB = YAB_list[i]
                S[i] = np.mean(YB * (YAB - YA)) / V if V > 0 else 0.0
                ST[i] = 0.5 * np.mean((YA - YAB) ** 2) / V if V > 0 else 0.0

            self.root.after(0, self.sobol_complete, names, S, ST, core, N_base)
        except Exception as e:
            self.root.after(0, self.sobol_error, str(e))

    def sobol_complete(self, names, S, ST, core, N_base):
        """Sobol complete"""
        self.is_running = False
        self.sobol_status.config(text="Complete ✓", foreground="green")
        self.sobol_progress.config(value=100)

        self.sobol_fig.clear()
        ax = self.sobol_fig.add_subplot(111)

        order = np.argsort(-ST)
        names_sorted = [names[i] for i in order]
        S_sorted = S[order]
        ST_sorted = ST[order]

        x = np.arange(len(names_sorted))
        ax.bar(x - 0.18, S_sorted, 0.36, label="S_i (First-order)", color='steelblue')
        ax.bar(x + 0.18, ST_sorted, 0.36, label="S_Ti (Total)", color='coral')

        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Sobol Index", fontsize=10)
        ax.set_title(f"Sobol Indices (core={core}, N={N_base})", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', ls='--', alpha=0.3)

        self.sobol_fig.tight_layout()
        self.sobol_canvas.draw()

        self.sobol_results = {"names": names_sorted, "S": S_sorted, "ST": ST_sorted}

    def sobol_error(self, msg):
        """Sobol error"""
        self.is_running = False
        self.sobol_status.config(text="Failed ✗", foreground="red")
        messagebox.showerror("Error", f"Sobol failed:\n{msg}")

    def update_sobol_from_baseline(self):
        """Update Sobol ranges based on current baseline values in FULL tab"""
        vals = self.get_values()
        if vals is None:
            return

        try:
            pct = float(self.sobol_pct.get()) / 100.0
        except ValueError:
            messagebox.showerror("Error", "Invalid uncertainty percentage")
            return

        for param, widgets in self.sobol_vars.items():
            if param in vals:
                baseline = float(vals[param])
                low = baseline * (1 - pct)
                high = baseline * (1 + pct)

                widgets["low"].delete(0, tk.END)
                widgets["low"].insert(0, f"{low:.4g}")

                widgets["high"].delete(0, tk.END)
                widgets["high"].insert(0, f"{high:.4g}")

        messagebox.showinfo("Success", f"Updated ranges with ±{pct*100:.1f}% uncertainty")

    # ========== EXPORT FUNCTIONS ==========
    def export_plot(self):
        """Export FULL plot"""
        if self.current_results is None:
            messagebox.showwarning("Warning", "No plot available")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def export_results(self):
        """Export FULL results"""
        if self.current_results is None:
            messagebox.showwarning("Warning", "No results")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if filename:
            with open(filename, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Saved to {filename}")

    def export_sens_plot(self):
        """Export SENS plot"""
        if not hasattr(self, 'sens_results'):
            messagebox.showwarning("Warning", "No SENS results")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filename:
            self.sens_fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def export_sobol_plot(self):
        """Export Sobol plot"""
        if not hasattr(self, 'sobol_results'):
            messagebox.showwarning("Warning", "No Sobol results")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filename:
            self.sobol_fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved to {filename}")

    def load_config(self):
        """Load config with validation"""
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)

                # ✅ Validate JSON schema
                expected_keys = set(self.entries.keys()) | {"core"}
                loaded_keys = set(config.keys())

                # Check for unknown keys
                unknown_keys = loaded_keys - expected_keys
                if unknown_keys:
                    response = messagebox.askyesno(
                        "Unknown Parameters",
                        f"Config contains unknown parameters:\n{', '.join(unknown_keys)}\n\n"
                        f"Continue loading known parameters?"
                    )
                    if not response:
                        return

                # Load valid parameters
                loaded_count = 0
                for key, value in config.items():
                    if key in self.entries:
                        self.entries[key].delete(0, tk.END)
                        self.entries[key].insert(0, str(value))
                        loaded_count += 1
                    elif key == "core":
                        if value in ["m2", "m3"]:
                            self.core_var.set(value)
                            loaded_count += 1

                messagebox.showinfo("Success", f"Loaded {loaded_count} parameters from config")

            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"Invalid JSON file:\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def save_config(self):
        """Save config"""
        vals = self.get_values()
        if vals is None:
            return
        vals["core"] = self.core_var.get()
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(vals, f, indent=2)
                messagebox.showinfo("Success", "Config saved")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_defaults(self):
        """Load defaults (already set in create_input_sections)"""
        pass


def main():
    root = tk.Tk()
    app = BucklingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

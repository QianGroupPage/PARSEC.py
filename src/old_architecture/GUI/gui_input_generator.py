import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import csv
import numpy as np

A0_ANG = 0.529177210903
DEFAULT_GRID_SPACING_ANG = 0.2
DEFAULT_RADIUS_ANG = 5.0

class InputGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parsec Python Input Generator")
        self.root.geometry("800x600")

        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- System Section (Left) ---
        system_frame = ttk.LabelFrame(main_frame, text="System (Atom Coordinates)", padding="5")
        system_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        system_label = ttk.Label(
            system_frame,
            text="Format: Element X Y Z\nExample:\nO 0.0 0.0 0.0\nH 0.0 1.43 1.1"
        )
        system_label.pack(anchor=tk.NW, pady=(0, 5))

        load_frame = ttk.Frame(system_frame)
        load_frame.pack(fill=tk.X, pady=(0, 5))

        load_btn = ttk.Button(load_frame, text="Load XYZ/POSCAR", command=self.load_structure_file)
        load_btn.pack(side=tk.LEFT)

        self.system_text = tk.Text(system_frame, width=40, height=20)
        self.system_text.pack(fill=tk.BOTH, expand=True)

        # --- Settings Section (Right) ---
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))

        # Variables
        self.unit_var = tk.StringVar(value="Angstrom")
        self.grid_spacing_var = tk.StringVar()
        self.radius_var = tk.StringVar()
        self.z_charge_var = tk.StringVar(value="0")
        self.density_method_var = tk.StringVar(value="SAD")
        self.sad_grid_var = tk.StringVar(value="default")
        self.ml_path_var = tk.StringVar()
        self.grid_poscar_path_var = tk.StringVar()
        self.tol_coeff_var = tk.StringVar(value="2")
        self.tol_exp_var = tk.StringVar(value="4")
        self.maxits_var = tk.StringVar(value="50")
        self.fermi_temp_var = tk.StringVar(value="500")
        self.save_wfn_var = tk.IntVar(value=0)
        self.use_gpu_var = tk.IntVar(value=0)
        self.recenter_atoms_var = tk.IntVar(value=1)
        self.fd_order_var = tk.StringVar(value="8")
        self.poldeg_var = tk.StringVar(value="10")
        self.diagmeth_var = tk.IntVar(value=3)
        self._current_unit = self.unit_var.get()
        self._element_order_map = None
        self._set_defaults_for_unit(self._current_unit)

        # Grid layout for settings
        row = 0

        # Unit
        ttk.Label(settings_frame, text="Unit :").grid(row=row, column=0, sticky=tk.W, pady=2)
        unit_combo = ttk.Combobox(settings_frame, textvariable=self.unit_var, state="readonly")
        unit_combo['values'] = ("Bohr", "Angstrom")
        unit_combo.grid(row=row, column=1, sticky=tk.EW, pady=2)
        unit_combo.bind("<<ComboboxSelected>>", self.on_unit_change)
        row += 1

        # Grid spacing
        ttk.Label(settings_frame, text="Grid Spacing h :").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.grid_spacing_entry = ttk.Entry(settings_frame, textvariable=self.grid_spacing_var)
        self.grid_spacing_entry.grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Sphere radius
        ttk.Label(settings_frame, text="Sphere Radius :").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.radius_entry = ttk.Entry(settings_frame, textvariable=self.radius_var)
        self.radius_entry.grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Z_charge
        ttk.Label(settings_frame, text="Z_charge (Net Charge) :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.z_charge_var).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Density Method
        ttk.Label(settings_frame, text="Density Method :").grid(row=row, column=0, sticky=tk.W, pady=2)
        density_combo = ttk.Combobox(settings_frame, textvariable=self.density_method_var, state="readonly")
        density_combo['values'] = ("SAD", "ML")
        density_combo.grid(row=row, column=1, sticky=tk.EW, pady=2)
        density_combo.bind("<<ComboboxSelected>>", self.update_density_ui)
        row += 1

        # SAD grid options (Conditional)
        self.sad_grid_label = ttk.Label(settings_frame, text="SAD Grid :")
        self.sad_grid_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.sad_grid_frame = ttk.Frame(settings_frame)
        self.sad_grid_frame.grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(
            self.sad_grid_frame,
            text="Default",
            variable=self.sad_grid_var,
            value="default",
            command=self.update_density_ui
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            self.sad_grid_frame,
            text="From .npy + POSCAR",
            variable=self.sad_grid_var,
            value="ml",
            command=self.update_density_ui
        ).pack(side=tk.LEFT)
        row += 1

        # ML/Grid File Path (Conditional)
        self.ml_label = ttk.Label(settings_frame, text="ML Density File (.npy):")
        self.ml_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ml_entry = ttk.Entry(settings_frame, textvariable=self.ml_path_var)
        self.ml_entry.grid(row=row, column=1, sticky=tk.EW, pady=2)
        self.ml_browse_btn = ttk.Button(settings_frame, text="Browse", command=self.browse_ml_file)
        self.ml_browse_btn.grid(row=row, column=2, padx=5, pady=2)
        row += 1

        self.grid_poscar_label = ttk.Label(settings_frame, text="Grid POSCAR File :")
        self.grid_poscar_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.grid_poscar_entry = ttk.Entry(settings_frame, textvariable=self.grid_poscar_path_var)
        self.grid_poscar_entry.grid(row=row, column=1, sticky=tk.EW, pady=2)
        self.grid_poscar_browse_btn = ttk.Button(settings_frame, text="Browse", command=self.browse_grid_poscar_file)
        self.grid_poscar_browse_btn.grid(row=row, column=2, padx=5, pady=2)
        row += 1

        # Tolerance
        ttk.Label(settings_frame, text="Tolerance :").grid(row=row, column=0, sticky=tk.W, pady=2)
        tol_frame = ttk.Frame(settings_frame)
        tol_frame.grid(row=row, column=1, sticky=tk.EW, pady=2)
        ttk.Entry(tol_frame, textvariable=self.tol_coeff_var, width=8).pack(side=tk.LEFT)
        ttk.Label(tol_frame, text=" x 10^(-").pack(side=tk.LEFT)
        ttk.Entry(tol_frame, textvariable=self.tol_exp_var, width=6).pack(side=tk.LEFT)
        ttk.Label(tol_frame, text=")").pack(side=tk.LEFT)
        row += 1

        # Fermi Temperature
        ttk.Label(settings_frame, text="Fermi Temp (K) :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.fermi_temp_var).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Max Iterations
        ttk.Label(settings_frame, text="Max Iterations :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.maxits_var).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Finite Difference Order
        ttk.Label(settings_frame, text="FD Order :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.fd_order_var).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Polynomial Degree
        ttk.Label(settings_frame, text="Polynomial Degree :").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(settings_frame, textvariable=self.poldeg_var).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1

        # Diagonalization Method
        ttk.Label(settings_frame, text="Diagonalization Method :").grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1

        diag_options = [
            (0, "0: Lanczos 1st step, then Chebyshev filtering"),
            (1, "1: Lanczos all the time"),
            (2, "2: Full-Chebyshev 1st step, then Chebyshev filtering"),
            (3, "3: Filter random vectors 1st step, then Chebyshev filtering")
        ]
        
        for val, text in diag_options:
            ttk.Radiobutton(settings_frame, text=text, variable=self.diagmeth_var, value=val).grid(row=row, column=0, columnspan=3, sticky=tk.W)
            row += 1

        ttk.Checkbutton(settings_frame, text="Save WFN file", variable=self.save_wfn_var).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=2
        )
        row += 1

        ttk.Checkbutton(
            settings_frame,
            text="Use GPU backend when available",
            variable=self.use_gpu_var,
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1

        self.recenter_atoms_btn = ttk.Checkbutton(
            settings_frame,
            text="Recenter atoms for default SAD grid",
            variable=self.recenter_atoms_var,
        )
        self.recenter_atoms_btn.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1

        # Generate Button
        generate_btn = ttk.Button(main_frame, text="Generate Input File", command=self.generate_file)
        generate_btn.pack(side=tk.BOTTOM, pady=10)

        # Initial state check
        self.update_density_ui()

    def update_density_ui(self, event=None):
        method = self.density_method_var.get().strip().lower()
        is_sad = method == "sad"
        use_ml_grid = self._is_ml_grid_active()

        sad_widgets = (self.sad_grid_label, self.sad_grid_frame)
        if is_sad:
            for widget in sad_widgets:
                widget.grid()
        else:
            for widget in sad_widgets:
                widget.grid_remove()

        if method == "ml":
            self.ml_label.config(text="ML Density/Grid File (.npy):")
        else:
            self.ml_label.config(text="Grid File (.npy):")

        ml_widgets = (
            self.ml_label,
            self.ml_entry,
            self.ml_browse_btn,
            self.grid_poscar_label,
            self.grid_poscar_entry,
            self.grid_poscar_browse_btn,
        )
        if use_ml_grid:
            for widget in ml_widgets:
                widget.grid()
        else:
            for widget in ml_widgets:
                widget.grid_remove()

        entry_state = "disabled" if use_ml_grid else "normal"
        self.grid_spacing_entry.configure(state=entry_state)
        self.radius_entry.configure(state=entry_state)

        recenter_state = "normal" if (is_sad and not use_ml_grid) else "disabled"
        self.recenter_atoms_btn.configure(state=recenter_state)

        if use_ml_grid:
            self.update_grid_from_files()

    def _is_ml_grid_active(self):
        method = self.density_method_var.get().strip().lower()
        return method == "ml" or (method == "sad" and self.sad_grid_var.get() == "ml")

    def browse_ml_file(self):
        default_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "npy")
        )
        if not os.path.isdir(default_dir):
            default_dir = os.getcwd()

        filename = filedialog.askopenfilename(
            title="Select .npy file",
            filetypes=[("Numpy files", "*.npy"), ("All files", "*.*")],
            initialdir=default_dir
        )
        if filename:
            self.ml_path_var.set(os.path.abspath(filename))
            self.update_grid_from_files()

    def browse_grid_poscar_file(self):
        filename = filedialog.askopenfilename(
            title="Select POSCAR file for grid",
            filetypes=[
                ("VASP/POSCAR files", "*.vasp *.poscar *.contcar"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.grid_poscar_path_var.set(os.path.abspath(filename))
            self.update_grid_from_files()

    def generate_file(self):
        # Gather data
        system_content = self.system_text.get("1.0", tk.END).strip()
        if not system_content:
            messagebox.showwarning("Missing Data", "Please enter atom coordinates in the System section.")
            return

        unit = "a.u." if self.unit_var.get() == "Bohr" else "Angstrom"
        z_charge = self.z_charge_var.get()
        density_choice = self.density_method_var.get().strip()
        density_method = density_choice.lower()
        ml_path = self.ml_path_var.get().strip()
        grid_poscar_path = self.grid_poscar_path_var.get().strip()
        use_ml_grid = self._is_ml_grid_active()
        if density_method == "sad" and self.sad_grid_var.get() == "ml":
            density_method = "sad_ml_grid"
        try:
            tol_value = self._compute_tolerance_value()
        except ValueError as exc:
            messagebox.showwarning("Invalid Tolerance", str(exc))
            return

        tol = f"{tol_value:.6e}"
        maxits = self.maxits_var.get()
        fermi_temp = self.fermi_temp_var.get()
        fd_order = self.fd_order_var.get()
        poldeg = self.poldeg_var.get()
        diagmeth = self.diagmeth_var.get()
        save_wfn = int(self.save_wfn_var.get())
        use_gpu = int(self.use_gpu_var.get())
        recenter_atoms = int(self.recenter_atoms_var.get())

        if use_ml_grid and not ml_path:
            messagebox.showwarning("Missing .npy file", "Please choose a .npy file for grid/density.")
            return
        if use_ml_grid and not grid_poscar_path:
            messagebox.showwarning("Missing POSCAR file", "Please choose a POSCAR file for grid settings.")
            return

        default_filename = self._build_default_filename(density_method, diagmeth)

        # Construct file content
        content = "$system\n"
        content += system_content + "\n"
        content += "$system\n\n"
        
        content += "$settings\n"
        content += f"unit = {unit}\n"
        content += f"Z_charge = {z_charge}\n"
        content += f"density_method = {density_method}\n"
        if use_ml_grid:
            ml_path = os.path.abspath(os.path.expanduser(ml_path))
            grid_poscar_path = os.path.abspath(os.path.expanduser(grid_poscar_path))
            content += f"ml_file_path = {ml_path}\n"
            content += f"grid_poscar_path = {grid_poscar_path}\n"
        else:
            grid_spacing = self.grid_spacing_var.get().strip()
            radius = self.radius_var.get().strip()
            if grid_spacing:
                content += f"grid_spacing = {grid_spacing}\n"
            if radius:
                content += f"sphere_radius = {radius}\n"
        content += f"tol = {tol}\n"
        if fermi_temp:
            content += f"Fermi_temp = {fermi_temp}\n"
        content += f"maxits = {maxits}\n"
        content += f"fd_order = {fd_order}\n"
        content += f"poldeg = {poldeg}\n"
        content += f"diagmeth = {diagmeth}\n"
        content += f"save_wfn = {save_wfn}\n"
        content += f"use_gpu = {use_gpu}\n"
        content += f"recenter_atoms = {recenter_atoms}\n"
        content += "$settings\n"

        # Save file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".in",
            filetypes=[("Input files", "*.in"), ("All files", "*.*")],
            initialfile=default_filename,
        )
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Input file saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def load_structure_file(self):
        filename = filedialog.askopenfilename(
            title="Select XYZ or POSCAR file",
            filetypes=[
                ("Structure files", "*.xyz *.vasp *.poscar *.contcar"),
                ("XYZ files", "*.xyz"),
                ("VASP/POSCAR files", "*.vasp *.poscar *.contcar"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return

        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".xyz":
                atoms = self.parse_xyz_file(filename)
            elif ext in (".vasp", ".poscar", ".contcar"):
                atoms = self.parse_poscar_file(filename)
            else:
                raise ValueError("Unsupported file type. Please select a .xyz or .vasp/.poscar/.contcar file.")

            self.system_text.delete("1.0", tk.END)
            self.system_text.insert(tk.END, "\n".join(atoms))
            self.unit_var.set("Angstrom")
            self.set_unit("Angstrom", preserve_values=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load structure file:\n{e}")

    def parse_xyz_file(self, path):
        with open(path, "r") as f:
            raw_lines = [line.rstrip("\n") for line in f]

        if not raw_lines:
            raise ValueError("Empty .xyz file.")

        start_idx = 0
        num_atoms = None
        try:
            num_atoms = int(raw_lines[0].split()[0])
            start_idx = 2
        except ValueError:
            start_idx = 0

        atoms = []
        for line in raw_lines[start_idx:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ line: '{line}'")
            element = parts[0]
            x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
            atoms.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
            if num_atoms is not None and len(atoms) >= num_atoms:
                break

        if not atoms:
            raise ValueError("No atom coordinates found in .xyz file.")
        if num_atoms is not None and len(atoms) < num_atoms:
            raise ValueError("XYZ file does not contain the expected number of atoms.")

        return atoms

    def _build_default_filename(self, density_method, diagmeth):
        formula = self._build_formula_from_system()
        if not formula:
            formula = "system"

        method_tag = self._density_method_tag(density_method)
        radius_ang, grid_ang = self._get_grid_values_ang()
        radius_str = self._format_filename_value(radius_ang)
        grid_str = self._format_filename_value(grid_ang)

        return f"{formula}_{method_tag}_diagmeth{diagmeth}_{radius_str}A_{grid_str}A"

    def _build_formula_from_system(self):
        system_content = self.system_text.get("1.0", tk.END).strip()
        if not system_content:
            return None

        counts = {}
        for raw_line in system_content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            symbol = parts[0]
            counts[symbol] = counts.get(symbol, 0) + 1

        if not counts:
            return None

        order_map = self._get_element_order_map()

        def sort_key(sym):
            return (order_map.get(sym, 10 ** 9), sym)

        pieces = []
        for symbol in sorted(counts.keys(), key=sort_key):
            count = counts[symbol]
            pieces.append(f"{symbol}{count}" if count != 1 else symbol)
        return "".join(pieces)

    def _get_element_order_map(self):
        if self._element_order_map is not None:
            return self._element_order_map

        element_order = []
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(base_dir, "elements_new.csv")
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r", newline="") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        if row and row[0].strip():
                            element_order.append(row[0].strip())
            except OSError:
                element_order = []

        self._element_order_map = {symbol: idx for idx, symbol in enumerate(element_order)}
        return self._element_order_map

    def _get_grid_values_ang(self):
        radius_val = self._parse_float(self.radius_var.get())
        grid_val = self._parse_float(self.grid_spacing_var.get())

        if radius_val is None:
            radius_val = self._default_value_for_unit(DEFAULT_RADIUS_ANG, self.unit_var.get())
        if grid_val is None:
            grid_val = self._default_value_for_unit(DEFAULT_GRID_SPACING_ANG, self.unit_var.get())

        if self.unit_var.get() == "Bohr":
            radius_val *= A0_ANG
            grid_val *= A0_ANG

        return radius_val, grid_val

    @staticmethod
    def _format_filename_value(value):
        formatted = f"{value:.6f}".rstrip("0").rstrip(".")
        if formatted == "":
            formatted = "0"
        return formatted.replace(".", "p")

    @staticmethod
    def _density_method_tag(density_method):
        if density_method == "sad_ml_grid":
            return "sadwithml"
        return density_method

    def parse_poscar_file(self, path):
        with open(path, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        if len(raw_lines) < 8:
            raise ValueError("Incomplete POSCAR file.")

        scale = float(raw_lines[1].split()[0])

        lattice = []
        for i in range(2, 5):
            parts = raw_lines[i].split()
            if len(parts) < 3:
                raise ValueError(f"Invalid lattice vector line: '{raw_lines[i]}'")
            lattice.append([float(parts[0]) * scale, float(parts[1]) * scale, float(parts[2]) * scale])

        idx = 5
        element_symbols = raw_lines[idx].split()
        if all(self._is_number(tok) for tok in element_symbols):
            raise ValueError("POSCAR file missing element symbols line.")
        idx += 1

        counts_line = raw_lines[idx].split()
        if not all(self._is_int(tok) for tok in counts_line):
            raise ValueError("Invalid atom counts line in POSCAR.")
        counts = [int(float(tok)) for tok in counts_line]
        idx += 1

        if len(counts) != len(element_symbols):
            raise ValueError("Element symbols count does not match atom counts in POSCAR.")

        coord_type_line = raw_lines[idx].lower()
        if coord_type_line.startswith("s"):
            idx += 1
            coord_type_line = raw_lines[idx].lower()

        if coord_type_line.startswith("d"):
            coord_type = "direct"
        elif coord_type_line.startswith("c"):
            coord_type = "cartesian"
        else:
            raise ValueError("Unknown coordinate type in POSCAR (expected Direct or Cartesian).")
        idx += 1

        num_atoms = sum(counts)
        coord_lines = raw_lines[idx:idx + num_atoms]
        if len(coord_lines) < num_atoms:
            raise ValueError("Not enough atom coordinates in POSCAR.")

        elements = []
        for symbol, count in zip(element_symbols, counts):
            elements.extend([symbol] * count)

        atoms = []
        for element, line in zip(elements, coord_lines):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid coordinate line: '{line}'")
            fx, fy, fz = (float(parts[0]), float(parts[1]), float(parts[2]))

            if coord_type == "direct":
                x = fx * lattice[0][0] + fy * lattice[1][0] + fz * lattice[2][0]
                y = fx * lattice[0][1] + fy * lattice[1][1] + fz * lattice[2][1]
                z = fx * lattice[0][2] + fy * lattice[1][2] + fz * lattice[2][2]
            else:
                x = fx * scale
                y = fy * scale
                z = fz * scale

            atoms.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")

        return atoms

    def update_grid_from_files(self):
        if not self._is_ml_grid_active():
            return

        npy_path = self.ml_path_var.get().strip()
        poscar_path = self.grid_poscar_path_var.get().strip()
        if not npy_path or not poscar_path:
            return
        if not os.path.isfile(npy_path) or not os.path.isfile(poscar_path):
            return

        try:
            density = np.load(npy_path, mmap_mode="r")
            if density.ndim != 3:
                raise ValueError("Grid .npy file must be a 3D array.")
            nx = density.shape[0]
            if nx <= 0:
                raise ValueError("Invalid grid shape in .npy file.")

            cube_length_ang = self.read_poscar_cube_length(poscar_path)
            h_ang = cube_length_ang / float(nx)
            radius_ang = cube_length_ang / 2.0

            if self.unit_var.get() == "Bohr":
                h_val = h_ang / A0_ANG
                radius_val = radius_ang / A0_ANG
            else:
                h_val = h_ang
                radius_val = radius_ang

            self.grid_spacing_var.set(self._format_float(h_val))
            self.radius_var.set(self._format_float(radius_val))
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to derive grid from files:\n{exc}")

    def read_poscar_cube_length(self, path):
        with open(path, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        if len(raw_lines) < 5:
            raise ValueError("Incomplete POSCAR file for grid.")

        scale = float(raw_lines[1].split()[0])
        lattice = []
        for i in range(2, 5):
            parts = raw_lines[i].split()
            if len(parts) < 3:
                raise ValueError(f"Invalid lattice vector line: '{raw_lines[i]}'")
            lattice.append([float(parts[0]) * scale, float(parts[1]) * scale, float(parts[2]) * scale])

        lengths = [self._vector_length(vec) for vec in lattice]
        if not lengths:
            raise ValueError("Could not determine lattice size from POSCAR.")

        return sum(lengths) / 3.0

    def _compute_tolerance_value(self):
        coeff = self._parse_float(self.tol_coeff_var.get())
        exp = self._parse_float(self.tol_exp_var.get())
        if coeff is None or exp is None:
            raise ValueError("Please enter numeric tolerance values.")
        if coeff <= 0:
            raise ValueError("Tolerance coefficient must be positive.")
        return coeff * (10 ** (-abs(exp)))

    def on_unit_change(self, event=None):
        self.set_unit(self.unit_var.get(), preserve_values=True)

    def set_unit(self, new_unit, preserve_values=True):
        old_unit = self._current_unit
        if old_unit == new_unit:
            return

        factor = self._unit_scale_factor(old_unit, new_unit)
        if preserve_values and factor is not None:
            grid_val = self._parse_float(self.grid_spacing_var.get())
            radius_val = self._parse_float(self.radius_var.get())
            if grid_val is not None:
                self.grid_spacing_var.set(self._format_float(grid_val * factor))
            else:
                self.grid_spacing_var.set(self._format_float(
                    self._default_value_for_unit(DEFAULT_GRID_SPACING_ANG, new_unit)
                ))
            if radius_val is not None:
                self.radius_var.set(self._format_float(radius_val * factor))
            else:
                self.radius_var.set(self._format_float(
                    self._default_value_for_unit(DEFAULT_RADIUS_ANG, new_unit)
                ))
        else:
            self._set_defaults_for_unit(new_unit)

        self._current_unit = new_unit
        if self._is_ml_grid_active():
            self.update_grid_from_files()

    def _set_defaults_for_unit(self, unit):
        self.grid_spacing_var.set(self._format_float(self._default_value_for_unit(DEFAULT_GRID_SPACING_ANG, unit)))
        self.radius_var.set(self._format_float(self._default_value_for_unit(DEFAULT_RADIUS_ANG, unit)))

    @staticmethod
    def _default_value_for_unit(value_ang, unit):
        return value_ang if unit == "Angstrom" else value_ang / A0_ANG

    @staticmethod
    def _format_float(value):
        return f"{value:.6f}"

    @staticmethod
    def _parse_float(value):
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _unit_scale_factor(old_unit, new_unit):
        if old_unit == "Bohr" and new_unit == "Angstrom":
            return A0_ANG
        if old_unit == "Angstrom" and new_unit == "Bohr":
            return 1.0 / A0_ANG
        return None

    @staticmethod
    def _is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_int(value):
        try:
            return float(value).is_integer()
        except ValueError:
            return False

    @staticmethod
    def _vector_length(vec):
        return (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5

if __name__ == "__main__":
    root = tk.Tk()
    app = InputGeneratorGUI(root)
    root.mainloop()

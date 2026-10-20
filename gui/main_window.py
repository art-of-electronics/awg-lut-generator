"""Tkinter interface for configuring, previewing, and exporting AWG waveforms."""
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as tkagg
import numpy as np

from core.modulation import combine
from core.waveforms import SAMPLE_COUNTS, WAVEFORM_NAMES, generate_waveform
from services.csv_export import discover_headers, export_lut, read_header_sample_count

FONT = ('Waree', 12)
CARRIER_DIVISORS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
OPERATIONS = ['Multiply', 'Add', 'Subtract']
ENVELOPE_REPETITIONS = {'1': ('1', 0), '2': ('2', 1), '4': ('4', 2)}
COLOR_TABLE = {
    'None': ['firebrick2', 'light gray', 'disabled', 'firebrick2', 'disabled', 'light gray'],
    'Sine': ['light gray', 'medium sea green', 'disabled', 'firebrick2', 'normal', 'medium sea green'],
    'Square': ['light gray', 'medium sea green', 'normal', 'light gray', 'normal', 'medium sea green'],
    'Exponential': ['light gray', 'medium sea green', 'normal', 'light gray', 'normal', 'medium sea green'],
    'Gaussian': ['light gray', 'medium sea green', 'normal', 'light gray', 'normal', 'medium sea green'],
    'Wave': ['light gray', 'medium sea green', 'disabled', 'firebrick2', 'normal', 'medium sea green'],
}


class AWGApplication:
    """Own the UI and mutable state for one AWG generator window."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize widgets, application state, and event callbacks.

        Args:
            root: Tkinter root window owned by the caller.
        """
        self.tk = root
        self.font=FONT
        self.colortab = COLOR_TABLE
        self.project_dir = Path(__file__).resolve().parents[1]
        self.headers_dir = self.project_dir / 'headers'
        self.output_dir = self.project_dir / 'output'
        self.headers = discover_headers(self.headers_dir)
        self.signal: np.ndarray | None = None
        self.filename = ''
        self.manual_samples = list(SAMPLE_COUNTS.keys())[4]
        self._previous_header = 'RAW'
        self._updating_header = False
        self.tk.geometry('640x695')
        self.tk.title('AWG LUT Generator')
        self.tk.attributes('-topmost', True)
        self.tk.resizable(False, False)
        self._create_widgets()
        self._sync_header_samples()
        self.generate()
        self._register_callbacks()

    def _create_widgets(self) -> None:
        """Create and position waveform controls and the preview canvas."""
        # Variables
        self.envelope_replicate_div = tk.IntVar(self.tk, value=1)
        self.envelope_par = tk.IntVar(self.tk)
        self.carrier_par = tk.IntVar(self.tk)

        # Menus
        self.carrier_div = tk.StringVar()
        self.carrier_div.set(CARRIER_DIVISORS[0])
        self.carrier_div_menu = tk.OptionMenu(self.tk, self.carrier_div, *CARRIER_DIVISORS)
        self.carrier_div_menu["font"] = self.font
        self.carrier_div_menu.grid(row=3, column=3, columnspan=3, sticky='nesw')
        self.carrier_div_menu["background"] = 'light gray'
        self.carrier_div_menu["state"] = 'disabled'

        self.function = tk.StringVar()
        self.function.set(OPERATIONS[0])
        self.function_menu = tk.OptionMenu(self.tk, self.function, *OPERATIONS)
        self.function_menu["font"] = self.font
        self.function_menu.grid(row=4, column=2, sticky='nesw')
        self.function_menu["background"] = 'light gray'
        self.function_menu["state"] = 'disabled'

        self.samples = tk.StringVar()
        self.samples.set(list(SAMPLE_COUNTS.keys())[4])
        self.samples_menu = tk.OptionMenu(self.tk, self.samples, *SAMPLE_COUNTS)
        self.samples_menu["font"] = self.font
        self.samples_menu.grid(row=5, column=2, sticky='nesw')
        self.samples_menu["background"] = 'light gray'
        self.samples_menu["activebackground"] = 'medium sea green'

        self.header = tk.StringVar()
        self.header.set(self.headers[0])
        self.header_menu = tk.OptionMenu(self.tk, self.header, *self.headers)
        self.header_menu["font"] = self.font
        self.header_menu.grid(row=5, column=3, columnspan=3, sticky='nesw')
        self.header_menu["background"] = 'light gray'
        self.header_menu["activebackground"] = 'medium sea green'


        # Labels
        tk.Label(self.tk, text='Envelope', font=self.font, width=15, relief="raised").grid(row=2, column=1, sticky='nesw')
        tk.Label(self.tk, text='Carrier', font=self.font, width=15, relief="raised").grid(row=3, column=1, sticky='nesw')
        tk.Label(self.tk, text='Function', font=self.font, width=15, relief="raised").grid(row=4, column=1, sticky='nesw')
        tk.Label(self.tk, text='Samples', font=self.font, width=15, relief="raised").grid(row=5, column=1, sticky='nesw')

        self.envelope_par_label = tk.Label(self.tk, text='0.1', font=self.font)
        self.envelope_par_label.grid(row=2, column=7, sticky='nesw')
        self.carrier_par_label = tk.Label(self.tk, text='0.1', font=self.font)
        self.carrier_par_label.grid(row=3, column=7, sticky='nesw')


        # Waves instances
        self.envelope = tk.StringVar()
        self.envelope.set(WAVEFORM_NAMES[0])
        self.envelope_menu = tk.OptionMenu(self.tk, self.envelope, *WAVEFORM_NAMES)
        self.envelope_menu.grid(row=2, column=2, sticky='nesw')
        self.envelope_menu["font"] = self.font
        self.envelope_menu["background"] = 'firebrick2'
        self.envelope_menu["activebackground"] = 'firebrick2'
        self.envelope_menu["width"] = 10

        self.carrier = tk.StringVar()
        self.carrier.set(WAVEFORM_NAMES[0])
        self.carrier_menu = tk.OptionMenu(self.tk, self.carrier, *WAVEFORM_NAMES)
        self.carrier_menu.grid(row=3, column=2, sticky='nesw')
        self.carrier_menu["font"] = self.font
        self.carrier_menu["background"] = 'firebrick2'
        self.carrier_menu["activebackground"] = 'firebrick2'
        self.carrier_menu["width"] = 10


        # Sliders
        self.envelope_slider = tk.Scale(self.tk, from_=1.0, to=9.0, orient='horizontal', showvalue=0, troughcolor='light gray',
                                      command=self.on_change, variable=self.envelope_par)
        self.envelope_slider.grid(row=2, column=6)
        self.envelope_slider["width"] = 30
        self.envelope_slider["length"] = 150
        self.envelope_slider["state"] = 'disabled'
        self.envelope_slider["troughcolor"] = 'firebrick2'

        self.carrier_slider = tk.Scale(self.tk, from_=1.0, to=9.0, orient='horizontal', showvalue=0,
                                      command=self.on_change, variable=self.carrier_par)
        self.carrier_slider.grid(row=3, column=6)
        self.carrier_slider["width"] = 30
        self.carrier_slider["length"] = 150
        self.carrier_slider["state"] = 'disabled'
        self.carrier_slider["troughcolor"] = 'firebrick2'


        # Buttons
        self.button_exit = tk.Button(self.tk, text="EXIT", command=self.exit, activebackground='firebrick2', font=self.font)
        self.button_exit.grid(row=7, column=1, columnspan=7, sticky='nesw')
        self.button_generate = tk.Button(self.tk, text="Generate LUT", state='disabled', command=self.export,
                                        activebackground='light gray', font=self.font)
        self.button_generate.grid(row=5, column=6, columnspan=2, sticky='nesw')

        self.radioButton = [None, None, None]
        for radio_text, (radio_value, radio_index) in ENVELOPE_REPETITIONS.items():
            self.radioButton[radio_index] = tk.Radiobutton(self.tk, text=radio_text,
                                                          variable=self.envelope_replicate_div, state='disabled',
                                                          value=radio_value, indicator=0, font=self.font,
                                                          background="medium slate blue", selectcolor="medium sea green",
                                                          command=self.on_change)
            self.radioButton[radio_index].grid(row=2, column=3 + radio_index, sticky='nesw')
            self.radioButton[radio_index]["width"] = 4


        # Plot
        figure = plt.Figure(dpi=100)
        self.ax = figure.add_subplot(111)
        self.plot = tkagg(figure, self.tk)
        self.plot.get_tk_widget().grid(row=6, column=1, columnspan=7, sticky='nesw')


    def _register_callbacks(self) -> None:
        """Connect Tkinter variable changes to the corresponding handlers."""
        for variable in (self.envelope, self.carrier, self.carrier_div,
                         self.function, self.samples):
            variable.trace_add('write', self.on_change)
        self.header.trace_add('write', self._on_header_change)

    def _sync_header_samples(self) -> None:
        """Apply a device header sample count or restore manual selection.

        Device-specific sample counts disable the sample selector. RAW
        and unrestricted headers restore the previous manual selection.
        """
        count = read_header_sample_count(self.header.get(), self.headers_dir)
        if count is None:
            self.samples.set(self.manual_samples)
            self.samples_menu['state'] = 'normal'
        else:
            label = next((key for key, value in SAMPLE_COUNTS.items()
                          if value == count), str(count))
            if label not in SAMPLE_COUNTS:
                # Device-specific lengths need not be powers of two.
                menu = self.samples_menu['menu']
                menu.add_command(label=label, command=tk._setit(self.samples, label))
            self.samples_menu['state'] = 'disabled'
            self.samples.set(label)

    def _on_header_change(self, *_args: object) -> None:
        """Handle header selection and recover from invalid templates."""
        if self._updating_header:
            return
        previous = self.header.get()
        self._updating_header = True
        try:
            self._sync_header_samples()
            self._previous_header = previous
        except (OSError, ValueError) as exc:
            self.header.set(self._previous_header)
            self._sync_header_samples()
            messagebox.showerror('Invalid device header', str(exc))
        finally:
            self._updating_header = False
        self.on_change()

    def _update_waveform_controls(self, name: str, is_envelope: bool) -> None:
        """Update widget colors and availability for a waveform type.

        Args:
            name: Selected waveform name.
            is_envelope: True for envelope controls, False for carrier.
        """
        colors = self.colortab[name]
        menu = self.envelope_menu if is_envelope else self.carrier_menu
        slider = self.envelope_slider if is_envelope else self.carrier_slider
        menu['background'] = colors[0]
        menu['activebackground'] = colors[1]
        slider['state'] = colors[2]
        slider['troughcolor'] = colors[3]
        if is_envelope:
            for button in self.radioButton:
                button['state'] = colors[4]
        else:
            self.carrier_div_menu['state'] = colors[4]
            self.carrier_div_menu['activebackground'] = colors[5]

    def on_change(self, *_args: object) -> None:
        """Refresh control states and regenerate the waveform preview."""
        envelope_name = self.envelope.get()
        carrier_name = self.carrier.get()
        self.envelope_par_label['text'] = str(self.envelope_par.get() / 10)
        self.carrier_par_label['text'] = str(self.carrier_par.get() / 10)
        enabled = envelope_name != 'None' or carrier_name != 'None'
        self.button_generate['state'] = 'normal' if enabled else 'disabled'
        self.button_generate['background'] = 'orange' if enabled else 'light gray'
        if enabled:
            self.button_generate['activebackground'] = 'medium sea green'
        self._update_waveform_controls(envelope_name, True)
        self._update_waveform_controls(carrier_name, False)
        both = envelope_name != 'None' and carrier_name != 'None'
        self.function_menu['state'] = 'normal' if both else 'disabled'
        self.function_menu['activebackground'] = 'medium sea green' if both else 'light gray'
        if self.samples_menu['state'] == 'normal':
            self.manual_samples = self.samples.get()
        self.generate()

    def generate(self) -> None:
        """Generate waveform samples, update the preview and CSV filename."""
        envelope_name = self.envelope.get()
        carrier_name = self.carrier.get()
        count = SAMPLE_COUNTS.get(self.samples.get(), int(self.samples.get()) if self.samples.get().isdigit() else 0)
        env, e_div, e_pv = generate_waveform(
            envelope_name, count, self.envelope_replicate_div.get(), self.envelope_par.get() / 10)
        car, c_div, c_pv = generate_waveform(
            carrier_name, count, int(self.carrier_div.get()), self.carrier_par.get() / 10)
        if envelope_name == 'None' and carrier_name == 'None':
            self.signal = None
            self.plot.draw()
            self.ax.clear()
            return
        self.signal, operator_suffix = combine(self.function.get(), env, car)
        # Draw before clearing the axes to preserve the preview behavior.
        self.ax.plot(self.signal, color='r')
        self.plot.draw()
        self.ax.clear()
        e_name = '' if envelope_name == 'None' else envelope_name.lower()
        c_name = '' if carrier_name == 'None' else carrier_name.lower()
        separator = '_'
        if envelope_name == 'None' or carrier_name == 'None':
            separator, operator_suffix = '', ''
        self.filename = f'{e_name}{separator}{c_name}{e_div}{e_pv}{operator_suffix}{c_div}{c_pv}.csv'

    def export(self) -> None:
        """Export the current waveform when samples are available."""
        if self.signal is not None:
            export_lut(self.signal, self.filename, self.header.get(),
                       self.output_dir, self.headers_dir)

    def exit(self) -> None:
        """Stop the Tkinter event loop."""
        self.tk.quit()


def run() -> None:
    """Create the AWG window and run its Tkinter event loop."""
    root = tk.Tk()
    AWGApplication(root)
    root.mainloop()
    root.destroy()

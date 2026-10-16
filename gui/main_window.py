"""Tkinter interface with instance-owned widgets and state (legacy layout preserved)."""
from pathlib import Path
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as tkagg
import numpy as np
from core.waveforms import SAMPLE_COUNTS, WAVEFORM_NAMES, generate_waveform
from core.modulation import combine
from services.csv_export import discover_headers, export_lut

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
        self.tk = root
        self.font = FONT
        self.colortab = COLOR_TABLE
        self.project_dir = Path(__file__).resolve().parents[1]
        self.headers_dir = self.project_dir / 'headers'
        self.output_dir = Path.cwd()
        self.headers = discover_headers(self.headers_dir)
        self.signal: np.ndarray | None = None
        self.filename = ''
        self.tk.geometry('640x695')
        self.tk.title('AWG LUT Generator')
        self.tk.attributes('-topmost', True)
        self.tk.resizable(False, False)
        self._create_widgets()
        self.generate()
        self._register_callbacks()

    def _create_widgets(self) -> None:
        #Variables
        self.envelope_replicate_div = tk.IntVar(self.tk, value=1)
        self.envelope_par = tk.IntVar(self.tk)
        self.carrier_par = tk.IntVar(self.tk)

        #Menus
        self.carrier_div = tk.StringVar()
        self.carrier_div.set(CARRIER_DIVISORS[0]) #Default value
        self.carrier_div_menu = tk.OptionMenu(self.tk, self.carrier_div, *CARRIER_DIVISORS)
        self.carrier_div_menu["font"] = self.font
        self.carrier_div_menu.grid(row=3, column=3, columnspan=3, sticky='nesw')
        self.carrier_div_menu["background"] = 'light gray'
        self.carrier_div_menu["state"] = 'disabled'

        #----
        self.function = tk.StringVar()
        self.function.set(OPERATIONS[0]) #Default value
        self.function_menu = tk.OptionMenu(self.tk, self.function, *OPERATIONS)
        self.function_menu["font"] = self.font
        self.function_menu.grid(row=4, column=2, sticky='nesw')
        self.function_menu["background"] = 'light gray'
        self.function_menu["state"] = 'disabled'
        #----

        self.samples = tk.StringVar()
        self.samples.set(list(SAMPLE_COUNTS.keys())[4]) #Default value
        self.samples_menu = tk.OptionMenu(self.tk, self.samples, *SAMPLE_COUNTS)
        self.samples_menu["font"] = self.font
        self.samples_menu.grid(row=5, column=2, sticky='nesw')
        self.samples_menu["background"] = 'light gray'
        self.samples_menu["activebackground"] = 'medium sea green'
        #----

        self.header = tk.StringVar()
        self.header.set(self.headers[0]) #Default value
        self.header_menu = tk.OptionMenu(self.tk, self.header, *self.headers)
        self.header_menu["font"] = self.font
        self.header_menu.grid(row=5, column=3, columnspan=3, sticky='nesw')
        self.header_menu["background"] = 'light gray'
        self.header_menu["activebackground"] = 'medium sea green'
        #----


        #Labels
        tk.Label(self.tk, text='Envelope', font=self.font, width=15, relief="raised").grid(row=2, column=1, sticky='nesw')
        tk.Label(self.tk, text='Carrier', font=self.font, width=15, relief="raised").grid(row=3, column=1, sticky='nesw')
        tk.Label(self.tk, text='Function', font=self.font, width=15, relief="raised").grid(row=4, column=1, sticky='nesw')
        tk.Label(self.tk, text='Samples', font=self.font, width=15, relief="raised").grid(row=5, column=1, sticky='nesw')

        self.envelope_par_label = tk.Label(self.tk, text='0.1', font=self.font)
        self.envelope_par_label.grid(row=2, column=7, sticky='nesw')
        self.carrier_par_label = tk.Label(self.tk, text='0.1', font=self.font)
        self.carrier_par_label.grid(row=3, column=7, sticky='nesw')


        #Waves instances
        self.envelope = tk.StringVar()
        self.envelope.set(WAVEFORM_NAMES[0]) #default value
        self.envelope_menu = tk.OptionMenu(self.tk, self.envelope, *WAVEFORM_NAMES)
        self.envelope_menu.grid(row=2, column=2, sticky='nesw')
        self.envelope_menu["font"] = self.font
        self.envelope_menu["background"] = 'firebrick2'
        self.envelope_menu["activebackground"] = 'firebrick2'
        self.envelope_menu["width"] = 10

        self.carrier = tk.StringVar()
        self.carrier.set(WAVEFORM_NAMES[0]) #default value
        self.carrier_menu = tk.OptionMenu(self.tk, self.carrier, *WAVEFORM_NAMES)
        self.carrier_menu.grid(row=3, column=2, sticky='nesw')
        self.carrier_menu["font"] = self.font
        self.carrier_menu["background"] = 'firebrick2'
        self.carrier_menu["activebackground"] = 'firebrick2'
        self.carrier_menu["width"] = 10


        #Sliders
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



        #Buttons
        self.button_exit = tk.Button(self.tk, text = "EXIT", command = self.exit, activebackground = 'firebrick2', font=self.font)
        self.button_exit.grid(row=7, column=1, columnspan=7, sticky='nesw')
        self.button_generate = tk.Button(self.tk, text = "Generate LUT", state = 'disabled', command = self.export,
                                        activebackground = 'light gray', font=self.font)
        self.button_generate.grid(row=5, column=6, columnspan=2, sticky='nesw')

        self.radioButton = [None, None, None]
        for radio_text, (radio_value, radio_index) in ENVELOPE_REPETITIONS.items():
            self.radioButton[radio_index] = tk.Radiobutton(self.tk, text = radio_text,
                                                          variable = self.envelope_replicate_div, state = 'disabled',
                                                          value = radio_value, indicator = 0, font = self.font,
                                                          background = "medium slate blue", selectcolor = "medium sea green",
                                                          command = self.on_change)
            self.radioButton[radio_index].grid(row=2, column=3+radio_index, sticky='nesw')
            self.radioButton[radio_index]["width"] = 4
        #----


        #Plot
        figure = plt.Figure(dpi=100)
        self.ax = figure.add_subplot(111)
        self.plot = tkagg(figure, self.tk)
        self.plot.get_tk_widget().grid(row=6, column=1, columnspan=7, sticky='nesw')
        #----



    def _register_callbacks(self) -> None:
        for variable in (self.envelope, self.carrier, self.carrier_div,
                         self.function, self.samples, self.header):
            variable.trace_add('write', self.on_change)

    def _update_waveform_controls(self, name: str, is_envelope: bool) -> None:
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
        self.generate()

    def generate(self) -> None:
        envelope_name = self.envelope.get()
        carrier_name = self.carrier.get()
        count = SAMPLE_COUNTS[self.samples.get()]
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
        # Legacy draw/clear order is intentional for identical rendering.
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
        if self.signal is not None:
            export_lut(self.signal, self.filename, self.header.get(),
                       self.output_dir, self.headers_dir)

    def exit(self) -> None:
        self.tk.quit()


def run() -> None:
    root = tk.Tk()
    app = AWGApplication(root)
    root.mainloop()
    root.destroy()

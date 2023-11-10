import pyvisa


class keithley2400:
    def __init__(self, channel=1, port=24):
        rsc = f"GPIB{channel}::{port}::INSTR"
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(rsc)

    def _id(self):
        id = self.instr.query('*IDN?')
        return id

    def vi_sweep(start, stop, step):
        voltages, currents = [], []
        return None

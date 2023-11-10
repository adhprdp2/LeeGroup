import pyvisa


class keithley2400:
    def __init__(self, channel=1, port=24):
        rsc = f"GPIB{channel}::{port}::INSTR"
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(rsc)

    def _id(self):
        id = self.instr.query('*IDN?')
        return id
    
    @property
    def voltage(self):
        reply = self.instr.query('*IDN?')
        return reply
    
    @voltage.setter
    def voltage(self, value: float):
        self._write(f'DISPLAY:WINDOW:TRACE:Y:RLEVEL {value} dBm')


    def vi_sweep(start, stop, step):
        voltages, currents = [], []
        return None

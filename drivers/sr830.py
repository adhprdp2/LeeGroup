import pyvisa


class sr830:
    def __init__(self, channel=1, port=15):
        rsc = f"GPIB{channel}::{port}::INSTR"
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(rsc)

    def _id(self):
        id = self.instr.query('*IDN?')
        return id

    def measure_impedance():
        return None

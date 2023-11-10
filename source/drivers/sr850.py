import pyvisa


class sr850:
    def __init__(self, channel=1, port=8):
        rsc = f"GPIB{channel}::{port}::INSTR"
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(rsc_name, read_termination='\n')
        self.instr.write_termination = '\n'

    def _id(self):
        id = self.instr.query('*IDN?', delay=1)
        return id

    def measure_impedance():
        return None

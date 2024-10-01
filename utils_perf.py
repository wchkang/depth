import usbtmc
import time

class Wt310device:
    def __init__(self, vendor_id=2849, product_id=37):
        self.instr = usbtmc.Instrument(vendor_id, product_id)

    def readvals(self):
        self.instr.write(":NUM:NORM:VAL?\n") 
        buf = self.instr.read()
        return buf.split(',')

    def sample(self):
        a = self.readvals()
        # default setting. to query, :NUM:NORM:ITEM1? for exampe
        # 1: Watt hour, 2: Current, 3: Active Power, 4: Apparent Power 
        # 5: Reactive Power, 6: Power factor
        # note that a's index is the item number minus one
	
	    #value dictionary
        ret = {}
        ret['WH'] = float(a[0])
        ret['J']  = float(a[0]) * 3600.0
        #ret['J']  = float(a[0]) * 1000
        ret['P']  = float(a[2])
        ret['PF'] = float(a[5])
        return ret

    def start(self):
        #  set the update interval  
        self.instr.write(":RATE 100MS") 
        # set item1 watt-hour
        self.instr.write(":NUM:NORM:ITEM1 WH,1")
        time.sleep(1)
        # start integration
        self.instr.write(":INTEG:MODE NORM")
        self.instr.write(":INTEG:TIM 2,0,0")  # maximum 2 hours 

	    # reset integration
        time.sleep(1)
        self.instr.write(":INTEG:RESET")

        time.sleep(1)
        self.instr.write(":INTEG:START")

    def stop(self):
        self.instr.write(":INTEG:STOP")
    
    def reset(self):
        self.instr.write('*RST')  # initialize the settings
        self.instr.write(':COMM:REM ON') # set remote mode

from pyxbos.process import run_loop, config_from_file
from pyxbos.drivers import pbc
import sys
import numpy as np
import warnings
import logging
import math
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(message)s')


class democontroller(pbc.LPBCProcess):

    def phasor_calc(self, local_phasors, reference_phasors, phase_channels):
        # Initialize
        local = [0] * len(phase_channels)
        ref = [0] * len(phase_channels)
        flag = [1] * len(phase_channels)

        # Extract latest 50 readings from uPMU
        for phase in range(len(phase_channels)):
            local[phase] = local_phasors[phase][-50:]
            ref[phase] = reference_phasors[phase][-50:]

        if np.size(self.Vmag_relative) == 0:
            # Initialize
            self.Vang_relative = np.empty((len(phase_channels), 1))
            self.Vmag_relative = np.empty((len(phase_channels), 1))
            # loops through every phase with actuation
            for phase in range(len(phase_channels)):
                # Initialize: Extract measurements from most recent timestamps only for first iteration
                V_mag_local = local[phase][-1]['magnitude']
                V_ang_local = local[phase][-1]['angle']
                V_mag_ref = ref[phase][-1]['magnitude']
                V_ang_ref = ref[phase][-1]['angle']

                self.Vang_relative[phase] = V_ang_local - V_ang_ref
                self.Vmag_relative[phase] = V_mag_local - V_mag_ref

        # loops through every phase with actuation
        for phase in range(len(local)):
            # loops through every local uPMU reading starting from most recent
            for local_packet in reversed(local[phase]):
                # extract most recent local uPMU reading and account for ametek delay
                local_time = int(local_packet['time']) - 50000
                # loops though every reference uPMU reading starting from most recent
                for ref_packet in reversed(ref[phase]):
                    ref_time = int(ref_packet['time'])

                    # check timestamps of local and reference uPMU if within 2 ms
                    if abs(ref_time - local_time) <= 2000000:
                        local_index = local[phase].index(local_packet)
                        ref_index = ref[phase].index(ref_packet)
                        # Extract measurements from closest timestamps
                        V_mag_local = local[phase][local_index]['magnitude']
                        V_ang_local = local[phase][local_index]['angle']
                        V_mag_ref = ref[phase][ref_index]['magnitude']
                        V_ang_ref = ref[phase][ref_index]['angle']

                        # calculates relative phasors
                        self.Vang_relative[phase] = V_ang_local - V_ang_ref
                        self.Vmag_relative[phase] = V_mag_local - V_mag_ref

                        flag[phase] = 0
                        break
                if flag[phase] == 0:
                    break
            if flag[phase] == 1:
                print("Phase", phase, ", Iteration ", self.iteration_counter,
                      ": No timestamp found. Using previous timestamps")

    def PQ_solver(self, local_phasors, phase_channels):
        # Initialize
        V_mag = [0] * len(phase_channels)
        V_ang = [0] * len(phase_channels)
        I_mag = [0] * len(phase_channels)
        I_ang = [0] * len(phase_channels)
        theta = [0] * len(phase_channels)

        if np.size(self.Pact) == 0:
            # Initialize
            self.Pact = np.empty((len(phase_channels), 1))
            self.Qact = np.empty((len(phase_channels), 1))

        "Make sure phases are in consecutive order in config. Voltage first, then current. i.e., L1, L2, I1, I2"
        for phase in range(len(phase_channels)):
            V_mag[phase] = local_phasors[phase][-1]['magnitude']
            V_ang[phase] = local_phasors[phase][-1]['angle']
            I_mag[phase] = local_phasors[(len(phase_channels) + phase + 1)][-1]['magnitude']  # check channels
            I_ang[phase] = local_phasors[(len(phase_channels) + phase + 1)][-1]['angle']  # check channels

            theta[phase] = V_ang[phase] - I_ang[phase]

            # P = (VI)cos(theta), Q = (VI)sin(theta)
            self.Pact[phase] = V_mag[phase] * I_mag[phase] * (math.cos(math.radians(theta[phase])))
            self.Qact[phase] = V_mag[phase] * I_mag[phase] * (math.sin(math.radians(theta[phase])))

    def __init__(self, cfg):
        super().__init__(cfg)

        # INITIALIZATION
        self.intError_mag = np.array([])
        self.intError_ang = np.array([])
        self.currentIntError_ang = np.array([])
        self.currentIntError_mag = np.array([])
        self.phasor_error_ang = np.array([])
        self.phasor_error_mag = np.array([])
        self.Pcmd = np.array([])
        self.Qcmd = np.array([])
        self.Pcmd_pu = np.array([])
        self.Qcmd_pu = np.array([])
        self.Pmax = np.array([])
        self.Qmax = np.array([])
        self.Psat = "initialize"
        self.Qsat = "initialize"
        self.ICDI_sigP = np.array([])
        self.ICDI_sigQ = np.array([])
        self.Vang_relative = np.array([])
        self.Vmag_relative = np.array([])
        self.Pact = np.array([])
        self.Qact = np.array([])
        self.local_index = np.array([])
        self.ref_index = np.array([])
        self.iteration_counter = 0
        self.Vang_targ = "initialize"
        self.Vmag_targ = "initialize"
        self.kvbase = np.array([])
        self.sbase = np.array([])
        self.Vang_relative_pu = np.array([])
        self.Vmag_relative_pu = np.array([])
        self.phase_channels = []
        self.phases = []

        #https config
        self.inv_id = 1 # # Change inverter id to unique inverter in HIL(lpbc number)
        self.pf_ctrl = 0.9
        self.inv_s_max = 7600

        # K gains
        self.Kp_ang = 0.068
        self.Ki_ang = 0.037
        self.Kp_mag = 3.8
        self.Ki_mag = 2.15

    def step(self, local_phasors, reference_phasors, phasor_target):
        self.iteration_counter += 1

        if phasor_target is None and self.Vang_targ == "initialize":
            print("Iteration", self.iteration_counter, ": No target received by SPBC")
            return

        else:
            if phasor_target is None:
                print("Iteration", self.iteration_counter, ": No target received by SPBC: Using last received target")

            else:
                "Data extractions"
                # extract out correct index of phasor target for each phase
                self.phases = phasor_target['phasor_targets'][0]['channelName']
                self.phase_channels = [0] * len(phasor_target['phasor_targets'])
                if len(self.phase_channels) > 1:
                    for i in range(len(self.phase_channels)):
                        self.phase_channels[i] = (phasor_target['phasor_targets'][i]['channelName'])
                    self.phases = self.phase_channels
                    if 'L1' in phase_channels:
                        for i, chan in enumerate(self.phase_channels):
                            if chan == 'L1':
                                self.phase_channels[i] = 0
                            if chan == 'L2':
                                self.phase_channels[i] = 1
                            if chan == 'L3':
                                self.phase_channels[i] = 2 - (3 - len(self.phase_channels))
                    else:
                        for i, chan in enumerate(self.phase_channels):
                            if chan == 'L2':
                                self.phase_channels[i] = 0
                            if chan == 'L3':
                                self.phase_channels[i] = 2 - (3 - len(self.phase_channels))
                    self.phases = sorted(self.phases)
                if self.Vang_targ == "initialize":
                    self.Vang_targ = np.empty((len(self.phase_channels), 1))
                    self.Vmag_targ = np.empty((len(self.phase_channels), 1))
                    self.kvbase = np.empty((len(self.phase_channels), 1))
                    self.sbase = np.empty((len(self.phase_channels), 1))

                # extract phasor target values for each phase
                for channel, phase in enumerate(phase_channels):
                    self.Vmag_targ[phase] = phasor_target['phasor_targets'][channel]['magnitude']
                    self.Vang_targ[phase] = phasor_target['phasor_targets'][channel]['angle']
                    self.kvbase[phase] = phasor_target['phasor_targets'][channel]['kvbase']['value']
                    self.sbase[phase] = phasor_target['phasor_targets'][channel]['kvabase']['value']

            # calculate relative voltage phasor
            self.phasor_calc(local_phasors, reference_phasors, self.phase_channels)

            # calculate P/Q from actuators
            self.PQ_solver(local_phasors, self.phase_channels)

            # convert to p.u.
            self.Vmag_relative_pu = self.Vmag_relative / (self.kvbase * 1000)
            self.Vang_relative_pu = self.Vang_relative / (self.kvbase * 1000)

            # calculate phasor errors
            self.phasor_error_ang = self.Vang_targ - self.Vang_relative_pu
            self.phasor_error_mag = self.Vmag_targ - self.Vmag_relative_pu

            if self.Psat == "initialize":
                n = 5  # saturation counter limit
                self.Psat = np.ones((np.size(self.phase_channels), n))
                self.Qsat = np.ones((np.size(self.phase_channels), n))
                self.ICDI_sigP = np.zeros((np.size(self.phase_channels), 1), dtype=bool)
                self.ICDI_sigQ = np.zeros((np.size(self.phase_channels), 1), dtype=bool)
                self.Pmax = np.empty((np.size(self.phase_channels), 1))
                self.Qmax = np.empty((np.size(self.phase_channels), 1))
                self.Pact = np.zeros((np.size(self.phase_channels), 1))
                self.Pcmd = np.zeros((np.size(self.phase_channels), 1))
                self.intError_ang = np.zeros((np.size(self.phase_channels), 1))
                self.intError_mag = np.zeros((np.size(self.phase_channels), 1))

            "Checking for P saturation (anti-windup control)"
            # find indicies where Pact + tolerance is less than Pcmd
            indexP = np.where(abs(self.Pact + (0.01 * self.Pcmd)) < abs(self.Pcmd))[0]

            # initialize saturation counter for each phase
            sat_arrayP = np.ones((np.size(self.phase_channels), 1))

            # stop integrator for saturated phases
            for i in indexP:
                sat_arrayP[i] = 0

            # saturation counter check to determine if ICDI signal should be sent to SPBC
            self.Psat = np.append(self.Psat, sat_arrayP, axis=1)
            self.Psat = self.Psat[:, 1:]

            for phase in range(len(self.phase_channels)):
                if phase in np.where(~self.Psat.any(axis=1))[0]:
                    self.ICDI_sigP[phase] = True
                    self.Pmax[phase] = self.Pact[phase]
                else:
                    self.ICDI_sigP[phase] = False
                    self.Pmax[phase] = None

            "Checking for Q saturation (anti-windup control)"
            # find indicies where Qact + tolerance is less than Qcmd
            indexQ = np.where(abs(self.Qact + (0.01 * self.Qcmd)) < abs(self.Qcmd))[0]

            # initialize saturation counter for each phase
            sat_arrayQ = np.ones((np.size(self.phase_channels), 1))

            # stop integrator for saturated phases
            for i in indexQ:
                sat_arrayQ[i] = 0

            # saturation counter check to determine if ICDI signal should be sent to SPBC
            self.Qsat = np.append(self.Qsat, sat_arrayQ, axis=1)
            self.Qsat = self.Qsat[:, 1:]

            for phase in range(len(self.phase_channels)):
                if phase in np.where(~self.Qsat.any(axis=1))[0]:
                    self.ICDI_sigQ[phase] = True
                    self.Qmax[phase] = self.Qact[phase]
                else:
                    self.ICDI_sigQ[phase] = False
                    self.Qmax[phase] = None

            "PI control algorithm"
            self.currentIntError_ang = (self.Ki_ang * self.phasor_error_ang) * sat_arrayP
            self.intError_ang += self.currentIntError_ang
            self.Pcmd_pu = (self.Kp_ang * self.phasor_error_ang) + self.intError_ang

            self.currentIntError_mag = (self.Ki_mag * self.phasor_error_mag) * sat_arrayQ
            self.intError_mag += self.currentIntError_mag
            self.Qcmd_pu = (self.Kp_mag * self.phasor_error_mag) + self.intError_mag

            # returns 8 signals: ICDI_sigP, ICDI_sigQ, Pmax, Qmax, phasor errors(2) to S-PBC; Pcmd, Qcmd to actuator
            # signals are column vector format: [number of phases/actuators x 1]

            # convert p.u. to W/ VARs (s base in units of kVA)
            self.Pcmd = self.Pcmd_pu * (self.sbase * 1000)
            self.Qcmd = self.Qcmd_pu * (self.sbase * 1000)

            "http to inverters"
            # Power triangles math
            inv_p_max = self.pf_ctrl * self.inv_s_max
            inv_q_max = math.sin(math.acos(self.pf_ctrl)) * self.inv_s_max

            # Change P/Q command as a percentage of inverter limit
            P_ctrl = (self.Pcmd / inv_p_max) * 100
            Q_ctrl = (self.Qcmd / inv_q_max) * 100

            #  Check hostname and port from Christoph
            #  Sends P and Q command to actuator
            for inv, P, Q in zip(self.inv_id, P_ctrl, Q_ctrl):
                requests.post(f"http://testhost.lbl.gov:1234/control?inv_id={inv},P_ctrl={P},pf_ctrl={self.pf_ctrl}")
                requests.post(f"http://testhost.lbl.gov:1234/control?inv_id={inv},Q_ctrl={Q},pf_ctrl={self.pf_ctrl}")

            "Status feedback to SPBC"
            status = {}
            status['phases'] = self.phases
            status['phasor_errors'] = {
                    'V': self.phasor_error_mag,
                    'delta': self.phasor_error_ang
                }
            status['p_saturated'] = self.ICDI_sigP
            status['q_saturated'] = self.ICDI_sigQ
            status['p_max'] = self.Pmax
            status['q_max'] = self.Qmax

            return status

if len(sys.argv) > 1:
    cfg = config_from_file(sys.argv[1])
else:
    sys.exit("Must supply config file as argument: python3 lpbc.py <config file.toml>")
lpbc1 = democontroller(cfg)
run_loop()

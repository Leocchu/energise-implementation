from pyxbos.process import run_loop, config_from_file
from pyxbos.drivers import pbc
import sys
import numpy as np
import warnings
import logging
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(message)s')


class democotroller(pbc.LPBCProcess):

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
        self.ametek_phase_shift = 0 # TODO: ametek phase shift in degrees

        # https config
        self.inv_id = 1  # TODO: Change inverter id/s to unique inverter in HIL(lpbc number): Must be in ascending order
        self.batt_max = 3300.
        self.inv_s_max = 7600.
        self.P_PV = np.array([])
        self.Pact_test = np.array([])
        self.batt_export = np.array([])
        self.batt_cmd = np.array([])
        self.mode = 0  # mode 1: PV as disturbance, mode 2: PV calculated, mode 3: PV only
        self.p_ctrl = np.array([])
        self.group_id = [0,1,2] # TODO: group id for motors corresponding to phase: Must be in ascending order
        self.local_s_ratio = 500/3.3 #TODO: Inverters should have ratio of 500/3.3 ; Load racks : 250
        self.Pcmd_inv = np.array([])
        self.Qcmd_inv = np.array([])

        # K gains
        self.Kp_ang = 0.068 # TODO
        self.Ki_ang = 0.037 # TODO
        self.Kp_mag = 3.8 # TODO
        self.Ki_mag = 2.15 #TODO

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
                V_ang_local = local[phase][-1]['angle'] - self.ametek_phase_shift
                V_mag_ref = ref[phase][-1]['magnitude']
                V_ang_ref = ref[phase][-1]['angle']

                self.Vang_relative[phase] = V_ang_local - V_ang_ref
                self.Vmag_relative[phase] = V_mag_local - V_mag_ref

        # loops through every phase with actuation
        for phase in range(len(local)):
            # loops through every local uPMU reading starting from most recent
            for local_packet in reversed(local[phase]):
                # extract most recent local uPMU reading
                local_time = int(local_packet['time'])
                # loops though every reference uPMU reading starting from most recent
                for ref_packet in reversed(ref[phase]):
                    ref_time = int(ref_packet['time'])

                    # check timestamps of local and reference uPMU if within 2 ms
                    if abs(ref_time - local_time) <= 2000000:
                        local_index = local[phase].index(local_packet)
                        ref_index = ref[phase].index(ref_packet)
                        # Extract measurements from closest timestamps
                        V_mag_local = local[phase][local_index]['magnitude']
                        V_ang_local = local[phase][local_index]['angle'] - self.ametek_phase_shift
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
                      ": No timestamp found")

    def PQ_solver(self, local_phasors, phase_channels):
        # Initialize
        V_mag = [0.0] * len(phase_channels)
        V_ang = [0.0] * len(phase_channels)
        I_mag = [0.0] * len(phase_channels)
        I_ang = [0.0] * len(phase_channels)
        theta = [0.0] * len(phase_channels)

        if np.size(self.Pact) == 0:
            # Initialize
            self.Pact = np.empty((len(phase_channels), 1))
            self.Qact = np.empty((len(phase_channels), 1))

        "Make sure phases are in consecutive order in config.  i.e., 123'/L1', 123'/L2, 123/C1, 123/C2, 123/L1, 123/L2" \
        "Model voltages first, then inverter currents, then inverter voltages (all in consecutive order)"
        for phase in range(len(phase_channels)):
            V_mag[phase] = local_phasors[phase + (2 * len(phase_channels)) + 1][-1]['magnitude']
            V_ang[phase] = local_phasors[phase + (2 * len(phase_channels)) + 1][-1]['angle']
            I_mag[phase] = local_phasors[(len(phase_channels) + phase + 1)][-1]['magnitude']  # check channels
            I_ang[phase] = local_phasors[(len(phase_channels) + phase + 1)][-1]['angle']  # check channels

            theta[phase] = V_ang[phase] - I_ang[phase]

            # P = (VI)cos(theta), Q = (VI)sin(theta)
            self.Pact[phase] = V_mag[phase] * I_mag[phase] * (np.cos(np.radians(theta[phase])))
            self.Qact[phase] = V_mag[phase] * I_mag[phase] * (np.sin(np.radians(theta[phase])))

    def step(self, local_phasors, reference_phasors, phasor_target):
        self.iteration_counter += 1

        if self.iteration_counter == 1:
            if self.mode == 1 or self.mode == 2:
                requests.get("http://131.243.41.47:9090/control?P_ctrl=97,Batt_ctrl=0")
            if self.mode == 3:
                requests.get("http://131.243.41.47:9090/control?P_ctrl=0,Batt_ctrl=0")
            self.Pact_test = np.zeros((len(phasor_target['phasor_targets']), 1))
            self.batt_export = np.ones((len(phasor_target['phasor_targets']), 1))
            self.batt_cmd = np.zeros((len(phasor_target['phasor_targets']), 1))
            self.p_ctrl = np.zeros((len(phasor_target['phasor_targets']), 1))

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
                    self.phases = [0] * len(self.phase_channels)
                    for i in range(len(self.phase_channels)):
                        self.phase_channels[i] = phasor_target['phasor_targets'][i]['channelName']
                        self.phases[i] = phasor_target['phasor_targets'][i]['channelName']
                    if 'L1' in self.phase_channels:
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

                # extract phasor target values for each phase: sorts in ascending phase order
                for channel, phase in enumerate(self.phase_channels):

                    self.Vmag_targ[phase] = phasor_target['phasor_targets'][channel]['magnitude']
                    self.Vang_targ[phase] = phasor_target['phasor_targets'][channel]['angle']
                    self.kvbase[phase] = phasor_target['phasor_targets'][channel]['kvbase']['value']
                    self.sbase[phase] = phasor_target['phasor_targets'][channel]['KVAbase']['value']

            # calculate relative voltage phasor
            self.phasor_calc(local_phasors, reference_phasors, self.phase_channels)

            # calculate P/Q from actuators
            self.PQ_solver(local_phasors, self.phase_channels)

            # convert to p.u.
            self.Vmag_relative_pu = self.Vmag_relative / (self.kvbase * 1000)

            # calculate phasor errors
            self.phasor_error_ang = self.Vang_targ - self.Vang_relative
            self.phasor_error_mag = self.Vmag_targ - self.Vmag_relative_pu

            if self.Psat == "initialize":
                n = 5  # saturation counter limit
                self.Psat = np.ones((np.size(self.phase_channels), n))
                self.Qsat = np.ones((np.size(self.phase_channels), n))
                self.ICDI_sigP = np.zeros((np.size(self.phase_channels), 1), dtype=bool)
                self.ICDI_sigQ = np.zeros((np.size(self.phase_channels), 1), dtype=bool)
                self.Pmax = np.empty((np.size(self.phase_channels), 1))
                self.Qmax = np.empty((np.size(self.phase_channels), 1))
                self.Pcmd = np.zeros((np.size(self.phase_channels), 1))
                self.Qcmd = np.zeros((np.size(self.phase_channels), 1))
                self.Pcmd_inv = np.zeros((np.size(self.phase_channels), 1))
                self.Qcmd_inv = np.zeros((np.size(self.phase_channels), 1))
                self.intError_ang = np.zeros((np.size(self.phase_channels), 1))
                self.intError_mag = np.zeros((np.size(self.phase_channels), 1))

            "Checking for P saturation (anti-windup control)"
            # find indicies where Pact + tolerance is less than Pcmd
            indexP = np.where(abs(self.Pact + (0.03 * self.Pcmd_inv)) < abs(self.Pcmd_inv))[0]

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
                    self.Pmax[phase] = self.Pact[phase] * self.local_s_ratio
                else:
                    self.ICDI_sigP[phase] = False
                    self.Pmax[phase] = None

            "Checking for Q saturation (anti-windup control)"
            # find indicies where Qact + tolerance is less than Qcmd
            indexQ = np.where(abs(self.Qact + (0.03 * self.Qcmd_inv)) < abs(self.Qcmd_inv))[0]

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
                    self.Qmax[phase] = self.Qact[phase] * self.local_s_ratio
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

            # convert p.u. to W/ VARs (s base in units of kVA)(Pcmd/Qcmd are values within ORT)
            self.Pcmd = self.Pcmd_pu * (self.sbase * 1000)
            self.Qcmd = self.Qcmd_pu * (self.sbase * 1000)

            "http to inverters"
            #  Check hostname and port
            #  Sends P and Q command to actuator
            if self.mode == 1: # PV disturbance
                self.Pcmd_inv = self.Pcmd / self.local_s_ratio
                self.Qcmd_inv = self.Qcmd / self.local_s_ratio
                self.P_PV = self.Pact - self.batt_export
                for phase, inv in zip(range(len(self.Pcmd)), self.inv_id):
                    self.batt_cmd[phase] = int(np.round(self.Pcmd[phase]))
                    if abs(self.batt_cmd[phase]) > 3300:
                        self.batt_cmd[phase] = int(np.sign(self.Pcmd[phase]) * 3300)
                    pf_ctrl = ((np.sign(self.Qcmd[phase]) * -1.0)*abs(self.Pcmd[phase])) /\
                              (np.sqrt((self.Pcmd[phase] ** 2) + (self.Qcmd[phase] ** 2)))
                    requests.get(f"http://131.243.41.47:9090/control?inv_id={inv},Batt_ctrl={self.batt_cmd[phase][0]},"
                                  f"pf_ctrl={pf_ctrl[0]}")
                    self.batt_export[phase] = self.batt_cmd[phase]

            if self.mode == 2: # PV subtracted
                self.Pcmd_inv = self.Pcmd / self.local_s_ratio
                self.Qcmd_inv = self.Qcmd / self.local_s_ratio
                self.P_PV = self.Pact - self.batt_export
                for phase, inv in zip(range(len(self.Pcmd)), self.inv_id):
                    self.batt_cmd[phase] = int(np.round(self.Pcmd_inv[phase] - self.P_PV[phase]))
                    if abs(self.batt_cmd[phase]) > 3300:
                        self.batt_cmd[phase] = int(np.sign(self.Pcmd_inv[phase]) * 3300)
                    pf_ctrl = ((np.sign(self.Qcmd_inv[phase]) * -1.0)*abs(self.Pcmd_inv[phase])) /\
                              (np.sqrt((self.Pcmd_inv[phase] ** 2) + (self.Qcmd_inv[phase] ** 2)))
                    requests.get(f"http://131.243.41.47:9090/control?inv_id={inv},Batt_ctrl={self.batt_cmd[phase][0]},"
                                  f"pf_ctrl={pf_ctrl[0]}")
                    self.batt_export[phase] = self.batt_cmd[phase]

            if self.mode == 3: # PV only
                self.Pcmd_inv = self.Pcmd / self.local_s_ratio
                self.Qcmd_inv = self.Qcmd / self.local_s_ratio
                for phase, inv in zip(range(len(self.Pcmd)), self.inv_id):
                    self.p_ctrl[phase] = int(np.round((abs(self.Pcmd_inv[phase]) / self.inv_s_max) * 100))
                    if self.p_ctrl[phase] > 97:
                        self.p_ctrl[phase] = 97
                        pf_ctrl = ((np.sign(self.Qcmd_inv[phase]) * -1.0) * abs(self.Pcmd_inv[phase])) / (self.inv_s_max * 0.97)
                    else:
                        pf_ctrl = ((np.sign(self.Qcmd_inv[phase]) * -1.0) * abs(self.Pcmd_inv[phase])) / \
                                  (np.sqrt((self.Pcmd_inv[phase] ** 2) + (self.Qcmd_inv[phase] ** 2)))
                    requests.get(f"http://131.243.41.47:9090/control?inv_id={inv},P_ctrl={self.p_ctrl[phase][0]},pf_ctrl={pf_ctrl[0]}")

            if self.mode == 4: # Load racks
                for phase, group in zip(range(len(self.Pcmd)), self.group_id):
                    self.p_ctrl[phase] = int(np.round((-1. * self.Pcmd_inv[phase]) + 1000))
                    if self.p_ctrl[phase] > 2000:
                        requests.get(f"http://131.243.41.118:9090/control?group_id={group},P_ctrl=2000")
                    elif self.p_ctrl < 0:
                        requests.get(f"http://131.243.41.118:9090/control?group_id={group},P_ctrl=0")
                    else:
                        requests.get(f"http://131.243.41.118:9090/control?group_id={group},P_ctrl={self.p_ctrl[phase][0]}")


            "Status feedback to SPBC"
            status = {}
            status['phases'] = self.phases
            status['phasor_errors'] = {
                    'V': list(self.phasor_error_mag.ravel()),
                    'delta': list(self.phasor_error_ang.ravel())
                }
            status['p_saturated'] = list(self.ICDI_sigP.ravel())
            status['q_saturated'] = list(self.ICDI_sigQ.ravel())
            status['p_max'] = list(self.Pmax.ravel())
            status['q_max'] = list(self.Qmax.ravel())

            return status

if len(sys.argv) > 1:
    cfg = config_from_file(sys.argv[1])
else:
    sys.exit("Must supply config file as argument: python3 lpbc.py <config file.toml>")
lpbc1 = democontroller(cfg)
run_loop()

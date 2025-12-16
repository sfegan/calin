#!/usr/bin/env python3

# calin/scripts/singles_curve.py -- Stephen Fegan - 2025-11-24
#
# Caclculate trogger delta-T times for given threhold / NSB rate.
#
# Note this accumualtes stats across multiple channele in parallel
# to take advantage of vectorization. The algorithm is therefore
# somewhat complicated by the need to track trigger times per channel.
#
# Copyright 2025, Stephen Fegan <sfegan@llr.in2p3.fr>
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris
#
# This file is part of "calin"
#
# "calin" is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 2 or later, as published by
# the Free Software Foundation.
#
# "calin" is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

import argparse
import json
import datetime
import platform
import numpy
import calin.simulation.detector_efficiency
import calin.simulation.vs_cta
import calin.simulation.ray_processor
import calin.iact_data.instrument_layout
import calin.iact_data.nectarcam_layout

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Bias curve generator')
parser.add_argument('-n', type=int, default=1000,
                   help='Specify the number of triggers to simulate')
parser.add_argument('--nchannels', type=int, default=16,
                    help='Specify the number of channels to use for the scalar rate curve')
parser.add_argument('--nsb', type=float, default=0.30,
                   help='Specify the NSB rate in GHz')
parser.add_argument('-t', '--threshold', type=float, default=120.0,
                   help='Specify the threshold in DC')
parser.add_argument('--noise', action='store_true', help='Add electronics noise')
parser.add_argument('-o', '--output', type=str, default=None,
                    help='Write trigger times to this file')
args = parser.parse_args()

nsb_rate = args.nsb
threshold = args.threshold
isample0 = 480

# Prepare a JSON header describing the run
config = vars(args).copy()
config['_generated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
config['_host'] = platform.node()
header = 'Configuration (JSON):\n' + json.dumps(config, indent=2)

def init():
    # Instantiate PE list processor
    global pe_list_processor
    pe_list_processor = calin.simulation.ray_processor.VCLWaveformPEProcessorFloat512(1,args.nchannels,1024,0.125,isample0)

    # Load and register impulse response
    pulse = numpy.loadtxt('Pulse_template_nectarCam_17042020-noshift.dat')
    hg = numpy.zeros(isample0)
    hg[0:len(pulse)] = pulse[:,1]
    hg[len(pulse):] = pulse[-1,1] * numpy.exp(-numpy.arange(isample0-len(pulse))/64)
    pe_list_processor.register_impulse_response(hg, 'DC')

    # Register camera response
    pe_list_processor.add_camera_response(numpy.asarray([0]),True)

    # Instantiate PE generator
    ap_pe_gen = calin.simulation.vs_cta.mstn_spe_and_afterpulsing_amplitude_generator(quiet=True)
    ap_pe_gen.this.disown() # Let pe_list_processor own it

    # Define NSB
    nsb = numpy.zeros(args.nchannels) + nsb_rate
    pe_list_processor.set_cr_nsb_rate(0, nsb, ap_pe_gen, True)

    # Define noise spectrum
    if(args.noise):
        def noise(f):
            fhi = 330.0
            flo = 0.0
            return (1+0.2*(f/275.0)**2)*(numpy.tanh((fhi-f)/20.0)+1)*0.9 + 0.6  # *(numpy.tanh((f-flo)/1.0)+1)/16
        freq = pe_list_processor.spectral_frequencies_ghz(False)
        noise_spectrum = noise(freq*1000)
        noise_spectrum[0] = 0
        noise_spectrum *= numpy.sqrt(10.5**2/16/pe_list_processor.noise_spectrum_var(noise_spectrum))
        pe_list_processor.set_cr_noise_spectrum(0, noise_spectrum)

    # Define threshold
    pe_list_processor.set_cr_threshold(0, numpy.zeros(args.nchannels) + threshold, 24)

def gen_triggers():
    all_ttrig = []
    ntrig = numpy.zeros(args.nchannels, dtype=int)
    ttrig = numpy.zeros(args.nchannels, dtype=int)

    pe_list_processor.clear_waveforms()
    pe_list_processor.add_nsb_noise_to_waveform_cr(0)
    pe_list_processor.convolve_impulse_response_fftw_codelet_cr(0)

    while(numpy.count_nonzero(ntrig<args.n) > 0):
        _, itrig = pe_list_processor.trigger_singles_cr(0, isample0)
    
        trigmask = (ttrig>=0) * (itrig>=isample0)
        ttrig[(ttrig>=0) * (itrig<isample0)] += 1024-isample0
        ttrig[ttrig<0] = 0
        ttrig[ntrig>=args.n] = -1
        for ichan in numpy.where(trigmask)[0]:
            all_ttrig.append(ttrig[ichan] + itrig[ichan] - isample0)
            ttrig[ichan] = -1
            ntrig[ichan] += 1

        pe_list_processor.shift_and_clear_waveforms(isample0)
        pe_list_processor.add_nsb_noise_to_waveform_cr(0, isample0)
        pe_list_processor.convolve_impulse_response_fftw_codelet_cr(0)
    return all_ttrig

trigger_times = []
init()
trigger_times = gen_triggers()

print(f'{len(trigger_times)} : {numpy.sum(trigger_times)/8e6:.2f} ms {1/numpy.mean(trigger_times)*8e6:.3f} kHz')

if args.output:
    numpy.savetxt(args.output, trigger_times, header=header, fmt='%d')
    print(f'Wrote {len(trigger_times)} trigger times to {args.output}')


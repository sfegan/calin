import os
import argparse
import concurrent
import pickle
import datetime
import platform
import numpy
import calin.math.geometry
import calin.ix.simulation.vcl_iact
import calin.simulation.tracker
import calin.simulation.vs_cta
import calin.simulation.ray_processor
import calin.simulation.world_magnetic_model
import calin.simulation.geant4_shower_generator
import calin.simulation.vcl_iact
import calin.iact_data.instrument_layout
import calin.iact_data.nectarcam_layout

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Nectarcam single-telescope shower trigger threshold calculation')
parser.add_argument('-n', type=int, default=1000,
                   help='Specify the number of showers to simulate')
parser.add_argument('--reuse', type=int, default=10,
                   help='Specify the number of times to reuse each shower')
parser.add_argument('--write_batch', type=int, default=100,
                   help='Specify the number of events to write per batch to output file')

parser.add_argument('-o', '--output', type=str, default=None,
                    help='Write trigger thresholds to this file')

parser.add_argument('--site', type=str, default='ctan', choices=['ctan','ctas'],
                    help='Site to simulate (default: ctan)')
parser.add_argument('-b', '--bmax', type=float, default=1000.0,
                   help='Specify the maximum shower impact parameter in meters')
parser.add_argument('--tts', type=float, default=0.75,
                   help='Specify the transit time spread RMS in ns')
parser.add_argument('--az', type=float, default=0.0,
                   help='Specify the telescope azimuth angle in degrees')
parser.add_argument('--el', type=float, default=70.0,
                   help='Specify the telescope elevation angle in degrees')
parser.add_argument('-e', '--energy', type=float, default=0.3,
                   help='Specify the telescope energy in TeV')
parser.add_argument('-p', '--primary', type=str, default='gamma', 
                    choices=['gamma','muon','electron','proton','helium','iron'],
                    help='Specify the primary particle type')
parser.add_argument('--viewcone', type=float, default=0.0,
                   help='Specify the random sampling offset between the primary direction and the telescope pointing direction in degrees')
parser.add_argument('--theta', type=float, default=0.0,
                   help='Specify the fixed offset between the primary direction and the telescope pointing direction in degrees')
parser.add_argument('--phi', type=float, default=0.0,
                   help='Specify the fixed polar angle of the primary direction around the telescope pointing direction in degrees')
parser.add_argument('--no_bfield', action='store_true', help='Disable the magnetic field (default: enabled)')


parser.add_argument('--nsb', type=float, default=0.30,
                   help='Specify the NSB rate in GHz')
parser.add_argument('--noise', action='store_true', help='Add electronics noise')
parser.add_argument('--no_after_pulsing', action='store_true', help='Disable after-pulsing in SPE spectrum (default: enabled)')
parser.add_argument('-t', '--trigger', type=str, default='3nn', choices=['3nn','4nn','m3','m4','multiplicity'],
                    help='Trigger algorithm to use (default: 3nn)')
parser.add_argument('-m', '--multiplicity', type=int, default=3,
                    help='Channel multiplicity if "multiplicity" algorithm is selected')
parser.add_argument('-c', '--coincidence', type=int, default=24,
                    help='Set the L1 trigger coincidence time in samples')

parser.add_argument('--threshold_min', type=float, default=40.0,
                    help='Minimum threshold for search (default: 40.0)')

parser.add_argument('--avx', type=int, default=512, choices=[128,256,512],
                    help='Set the AVX vector size in bits (default: 512)')

args = parser.parse_args()

tcoincidence = args.coincidence
iact = None

# Prepare a JSON header describing the run
config = vars(args).copy()
config['_generated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
config['_host'] = platform.node()

def init():
    numpy.random.seed()

    # Select simulation classes based on AVX size requested
    if args.avx == 128:
        iact_class = calin.simulation.vcl_iact.VCLIACTArray128
        electronics_sim_class = calin.simulation.ray_processor.VCLWaveformPEProcessorFloat128 
    elif args.avx == 256:
        iact_class = calin.simulation.vcl_iact.VCLIACTArray256
        electronics_sim_class = calin.simulation.ray_processor.VCLWaveformPEProcessorFloat256
    else:
        iact_class = calin.simulation.vcl_iact.VCLIACTArray512
        electronics_sim_class = calin.simulation.ray_processor.VCLWaveformPEProcessorFloat512

    # Load camera layout and reoder it by spiral channel index
    global nchan
    ncam = calin.iact_data.nectarcam_layout.nectarcam_layout()
    scam = calin.iact_data.instrument_layout.reorder_camera_channels(ncam, ncam.pixel_spiral_channel_index())
    nchan = scam.channel_size()

    # Load site-specific atmosphere, observation level and array layout
    global zobs
    global atm
    global atm_abs
    global mst
    global nscope
    if args.site == 'ctan':
        zobs = calin.simulation.vs_cta.ctan_observation_level()
        atm = calin.simulation.vs_cta.ctan_atmosphere(quiet=True)
        atm_abs = calin.simulation.vs_cta.ctan_atmospheric_absorption(quiet=True)
        mst = calin.simulation.vs_cta.mstn1_config()
    else:
        zobs = calin.simulation.vs_cta.ctas_observation_level()
        atm = calin.simulation.vs_cta.ctas_atmosphere(quiet=True)
        atm_abs = calin.simulation.vs_cta.ctas_atmospheric_absorption(quiet=True)
        mst = calin.simulation.vs_cta.msts1_config()
    mst.mutable_prescribed_array_layout().mutable_scope_positions(0).set_x(0)
    mst.mutable_prescribed_array_layout().mutable_scope_positions(0).set_y(0)
    nscope = mst.prescribed_array_layout().scope_positions_size()

    # Configure IACT array
    global iact
    cfg = iact_class.default_config()
    cfg.set_refraction_mode(calin.ix.simulation.vcl_iact.REFRACT_ONLY_CLOSE_RAYS)
    iact = iact_class(atm, atm_abs, cfg)

    # Load detector and efficiency models and the SPE generator
    global det_eff
    global cone_eff
    global pe_gen
    det_eff = calin.simulation.vs_cta.mstn_detection_efficiency(quiet=True)
    cone_eff = calin.simulation.vs_cta.mstn_cone_efficiency(quiet=True)
    pe_gen = calin.simulation.vs_cta.mstn_spe_amplitude_generator(quiet=True)

    # Add telescope arrays
    global all_pe_processor
    global all_prop
    all_pe_processor = []
    all_prop = []
    for i in range(max(1, args.reuse)):
        iact.add_propagator_set(args.bmax*100, f"Super array {i}")
        pe_processor = calin.simulation.ray_processor.SimpleListPEProcessor(nscope,nchan)
        prop = iact.add_davies_cotton_propagator(mst, pe_processor, det_eff, cone_eff, pe_gen, args.tts, 'MSTN')
        all_pe_processor.append(pe_processor)
        all_prop.append(prop)

    # Set telescope pointing direction
    global pt_dir
    iact.point_all_telescopes_az_el_deg(args.az, args.el)
    el = args.el * numpy.pi/180.0
    az = args.az * numpy.pi/180.0
    pt_dir  = numpy.asarray([numpy.cos(el)*numpy.sin(az), numpy.cos(el)*numpy.cos(az), numpy.sin(el)])

    # Magnetic field (if not disabled)
    global bfield
    if args.no_bfield:
        bfield = None
    else:
        wmm = calin.simulation.world_magnetic_model.WMM()
        bfield = wmm.field_vs_elevation(mst.array_origin().latitude(), mst.array_origin().longitude())

    # Configure Geant4 shower generator
    cfg = calin.simulation.geant4_shower_generator.Geant4ShowerGenerator.customized_config(
        1000, 0, atm.top_of_atmosphere(), calin.simulation.geant4_shower_generator.VerbosityLevel_SUPRESSED_STDOUT)
    cfg.add_pre_init_commands('/process/msc/StepLimit UseDistanceToBoundary')
    cfg.add_pre_init_commands('/process/msc/StepLimitMuHad UseDistanceToBoundary')
    cfg.add_pre_init_commands('/process/msc/RangeFactor 0.001')
    # cfg.add_pre_init_commands('/process/em/verbose 1')

    # Instantiate Geant4 shower generator
    global generator
    generator = calin.simulation.geant4_shower_generator.Geant4ShowerGenerator(atm, cfg, bfield);
    generator.set_minimum_energy_cut(20); # 20 MeV cut on KE (e-,p+,n,ions) or Etot

    # Load impulse response
    global isample0
    global dtsample
    global nsample
    pulse = calin.simulation.vs_cta.mstn_impulse_response()
    hg = pulse['hg']
    dtsample = pulse['dt']
    isample0 = len(hg)
    nsample = 1 << (len(hg)-1).bit_length() + 1  # Next power of two greater than 2*len(hg)

    # Instantiate electronics simulation
    global electronics_sim
    electronics_sim = electronics_sim_class(1,scam.channel_size(),nsample,dtsample,isample0)

    # Register impulse response
    electronics_sim.register_impulse_response(hg, 'DC')

    # Register camera response
    electronics_sim.add_camera_response(numpy.asarray([0]),True)

    # Configure the neighbors matrix
    electronics_sim.set_cr_neighbors(0, scam)

    # Select trigger algorithm
    global trigger_method
    if args.trigger == 'multiplicity':
        trigger_method = 'trigger_multiplicity_cr'
        electronics_sim.set_cr_multiplicity(0, args.multiplicity)
    elif args.trigger == 'm3':
        trigger_method = 'trigger_multiplicity_cr'
        electronics_sim.set_cr_multiplicity(0, 3)
    elif args.trigger == 'm4':
        trigger_method = 'trigger_multiplicity_cr'
        electronics_sim.set_cr_multiplicity(0, 4)
    elif args.trigger == '3nn':
        trigger_method = 'trigger_3nn_cr'
    elif args.trigger == '4nn':
        trigger_method = 'trigger_4nn_cr'
    else:
        raise ValueError(f'Unknown trigger algorithm: {args.trigger}')

    # Instantiate PE generator
    pe_gen_nsb = None
    if args.no_after_pulsing:
        pe_gen_nsb = calin.simulation.vs_cta.mstn_spe_amplitude_generator(quiet=True)
    else:
        pe_gen_nsb = calin.simulation.vs_cta.mstn_spe_and_afterpulsing_amplitude_generator(quiet=True)
    pe_gen_nsb.this.disown() # Let electronics_sim own it

    # Define NSB
    if args.nsb>0:
        nsb = numpy.zeros(scam.channel_size()) + args.nsb
        electronics_sim.set_cr_nsb_rate(0, nsb, pe_gen_nsb, True)

    # Define noise spectrum
    if(args.noise):
        def noise(f):
            fhi = 330.0
            flo = 0.0
            return (1+0.2*(f/275.0)**2)*(numpy.tanh((fhi-f)/20.0)+1)*0.9 + 0.6  # *(numpy.tanh((f-flo)/1.0)+1)/16
        freq = electronics_sim.spectral_frequencies_ghz(False)
        noise_spectrum = noise(freq*1000)
        noise_spectrum[0] = 0
        noise_spectrum *= numpy.sqrt(10.5**2/16/electronics_sim.noise_spectrum_var(noise_spectrum))
        electronics_sim.set_cr_noise_spectrum(0, noise_spectrum)

def gen_event():
    e = args.energy * 1e6 # Convert TeV to MeV

    costheta = 1.0 - (1.0 - numpy.cos(args.viewcone * numpy.pi/180))*numpy.random.uniform()
    theta = numpy.arccos(costheta)
    phi = numpy.random.uniform() * 2*numpy.pi
    u = numpy.asarray([numpy.sin(theta)*numpy.cos(phi), numpy.sin(theta)*numpy.sin(phi), numpy.cos(theta)])

    theta = args.theta * numpy.pi/180
    phi = args.phi * numpy.pi/180
    v = numpy.asarray([numpy.sin(theta)*numpy.cos(phi), numpy.sin(theta)*numpy.sin(phi), numpy.cos(theta)])
    calin.math.geometry.rotate_in_place_z_to_u_Rzy(u, v)
    calin.math.geometry.rotate_in_place_z_to_u_Rzy(u, -pt_dir)

    x0 = numpy.asarray([0,0,atm.zobs(0)]) + u/u[2]*(atm.top_of_atmosphere() - atm.zobs(0))
    if args.primary == 'gamma':
        pt = calin.simulation.tracker.ParticleType_GAMMA
    elif args.primary == 'muon':
        pt = calin.simulation.tracker.ParticleType_MUON
    elif args.primary == 'electron':
        pt = calin.simulation.tracker.ParticleType_ELECTRON
    elif args.primary == 'proton':
        pt = calin.simulation.tracker.ParticleType_PROTON
    elif args.primary == 'helium':
        pt = calin.simulation.tracker.ParticleType_HELIUM
    elif args.primary == 'iron':
        pt = calin.simulation.tracker.ParticleType_IRON
    else:
        raise ValueError(f'Unknown primary particle type: {args.primary}')

    generator.generate_showers(iact, 1, pt, e, x0, u)

    return e,pt,u,x0,costheta

def find_threshold(iarray):
    # Clear previous waveforms, transfer the PEs, add NSB, convolve impulse response
    electronics_sim.clear_waveforms()
    electronics_sim.transfer_scope_pes_to_waveform(all_pe_processor[iarray], 0)
    if args.nsb>0:
        electronics_sim.add_nsb_noise_to_waveform_cr(0)
    electronics_sim.convolve_impulse_response_fftw_codelet_cr(0)

    # Now perform threshold search
    trigger_fn = getattr(electronics_sim, trigger_method)
    
    # Start at threshold_min
    threshold = args.threshold_min
    electronics_sim.set_cr_threshold(0, numpy.zeros(nchan) + threshold, tcoincidence)
    itrig = trigger_fn(0, isample0)
    if itrig == -1:
        return -1  # No trigger even at min threshold
    
    # Double until it fails, updating lower bound
    lower = threshold
    upper = threshold
    while True:
        upper *= 2
        electronics_sim.set_cr_threshold(0, numpy.zeros(nchan) + upper, tcoincidence)
        itrig = trigger_fn(0, isample0)
        if itrig == -1:
            break
        lower = upper
    while (upper - lower) / lower > 0.01:
        mid = (lower + upper) / 2
        electronics_sim.set_cr_threshold(0, numpy.zeros(nchan) + mid, tcoincidence)
        itrig = trigger_fn(0, isample0)
        if itrig == -1:
            upper = mid
        else:
            lower = mid
    
    return lower

def one_event():
    event_results = []
    try:
        e,pt,u,x0,costheta = gen_event()
        for iarray in range(args.reuse):
            threshold = find_threshold(iarray)
            event_results.append(dict(
                iarray         = iarray,
                e              = e,
                pt             = int(pt),
                u0             = u.tolist(),
                x0             = x0.tolist(),
                costheta       = costheta,
                b              = iact.scattered_distance(iarray),
                offset         = iact.scattered_offset(iarray)[0:2].tolist(),
                threshold      = threshold))
    except Exception as ex:
        print(f'Error simulating event: {ex}')
        raise
    return event_results

def save_results(results, filehandle):
    output = dict(
        config = config,
        results = results)
    pickle.dump(output, filehandle)

all_results = []
max_workers = os.cpu_count() or 4
batch_size = max_workers * 10
with open(args.output, 'wb') as f:
    with concurrent.futures.ProcessPoolExecutor(initializer=init, max_workers=max_workers) as executor:
        remaining = args.n
        futures = set()

        # Submit initial batch
        first_batch = min(batch_size, remaining)
        for _ in range(first_batch):
            futures.add(executor.submit(one_event))
        remaining -= first_batch

        # Keep a bounded number of in-flight futures; as one completes, submit another
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                futures.remove(fut)
                for r  in fut.result():
                    all_results.append(r)

                if args.output and len(all_results)>0 and (len(all_results)%args.write_batch)==0:
                    save_results(all_results[len(all_results)-args.write_batch:], f)
                    print(f'Wrote {len(all_results)} results to {args.output}')

                if remaining > 0:
                    futures.add(executor.submit(one_event))
                    remaining -= 1

if args.output:
    # Re-write the output file as one single pickle rather than multiple appends
    with open(args.output, 'wb') as f:
        save_results(all_results, f)
        print(f'Wrote {len(all_results)} results to {args.output}')

th = args.threshold_min
while True:
    filtered_results = [r for r in all_results if r['threshold'] >= th]
    ntrig = len(filtered_results)
    if(ntrig < 10 or th > 10*config['threshold_min']):
        break
    if(ntrig < 0.9*len(all_results)):
        bmax = percentile([r['b'] for r in filtered_results],pc)
        thmax = percentile([arccos(r['costheta'])/pi*180.0 for r in filtered_results],pc)        
        print(f'{th:6.1f} {th/20:6.2f} | {ntrig:6d} {ntrig/len(all_results):5.3f} | {ntrig/len(all_results)*numpy.pi*bsim**2:.3e} {bmax*1e-2:6.1f} | {thmax:6.4f}')
    th = th + 10.0

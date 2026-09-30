# calin/python/simulation/vs_panoseti.py -- Stephen Fegan -- 2026-09-24
#
# Functions for returning instances of simulation configurations for PANOSETI arrays
#
# Copyright 2026, Stephen Fegan <sfegan@llr.in2p3.fr>
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

import os
import json
import numpy

import calin.simulation.atmosphere
import calin.simulation.detector_efficiency
import calin.math.spline_interpolation
import calin.ix.simulation.panoseti_optics
import calin.provenance.system_info
import calin.provenance.chronicle

def ds_filename(filename):
    if os.path.exists(filename):
        return filename
    
    # Try installed data directory
    try:
        data_dir = calin.provenance.system_info.build_info().data_install_dir() + "/simulation/"
        installed_path = os.path.join(data_dir, filename)
        if os.path.exists(installed_path):
            return installed_path
    except Exception:
        pass

    # Try relative to current working directory
    cwd_path = os.path.join(os.getcwd(), 'data', 'simulation', filename)
    if os.path.exists(cwd_path):
        return os.path.abspath(cwd_path)

    # Try repository local data directory relative to this file
    for rel_prefix in ['../../data/simulation', '../../../../data/simulation', '../../../data/simulation']:
        repo_data = os.path.join(os.path.dirname(__file__), rel_prefix, filename)
        if os.path.exists(repo_data):
            return os.path.abspath(repo_data)

    return filename

def dms(d, m, s):
    # Note that "negative" d=0 (e.g. -00:30:00) must be specified as 00:-30:00 or 00:00:-30
    sign = 1
    if d < 0:
        sign = -1
        d = abs(d)
    elif d == 0 and m < 0:
        sign = -1
        m = abs(m)
    elif d == 0 and m == 0 and s < 0:
        sign = -1
        s = abs(s)
    return sign * (d + m / 60.0 + s / 3600.0)

def palomar_observation_level(level_km = 1.61585):
    return level_km * 1e5

def palomar_location():
    # Palomar Observatory coordinates
    # Lat: 33°21'23.76"N 116°51'51.05"W
    lat_deg = dms(33, 21, 23.76)
    lon_deg = dms(-116, 51, 51.05)
    alt_cm = palomar_observation_level()
    return lat_deg, lon_deg, alt_cm

def palomar_atmosphere(profile = 'ecmwf_intermediate', standard_profiles = {
            'ecmwf_intermediate': 'atmprof_ecmwf_north_intermediate_fixed.dat',
            'ecmwf_summer': 'atmprof_ecmwf_north_summer_fixed.dat',
            'ecmwf_winter': 'atmprof_ecmwf_north_winter_fixed.dat' },
        zobs = palomar_observation_level(), quiet = False):
    cfg = calin.simulation.atmosphere.LayeredRefractiveAtmosphere.default_config()
    cfg.set_angular_model_optimization_altitude(18e5)
    cfg.set_zn_reference(45)
    cfg.set_zn_optimize([5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60])
    cfg.set_high_accuracy_mode(True)
    profile_path = standard_profiles[profile] if profile in standard_profiles else profile
    atm = calin.simulation.atmosphere.LayeredRefractiveAtmosphere(ds_filename(profile_path), zobs, cfg)

    if not quiet:
        print('Loading atmospheric profile :', profile_path)
        print('Observation level : %.3f km, thickness %.1f g/cm^2' % (
            atm.zobs(0) * 1e-5, atm.thickness(atm.zobs(0))))
        print('Cherenkov angle at %.3f, 10, 20 km : %.3f, %.3f, %.3f deg' % (
            atm.zobs(0) * 1e-5,
            numpy.arccos(1 / (atm.n_minus_one(atm.zobs(0)) + 1)) / numpy.pi * 180,
            numpy.arccos(1 / (atm.n_minus_one(10e5) + 1)) / numpy.pi * 180,
            numpy.arccos(1 / (atm.n_minus_one(20e5) + 1)) / numpy.pi * 180))
        prop_ct = atm.propagation_ct_correction(atm.top_of_atmosphere()) - atm.propagation_ct_correction(atm.zobs(0))
        print('Vertical propagation delay from %d to %.3f km : %.2f ns (%.1f cm)' % (
            atm.top_of_atmosphere() * 1e-5, atm.zobs(0) * 1e-5, prop_ct * 0.03335641, prop_ct))

    return atm

def palomar_atmospheric_absorption(absorption_model = 'navy_maritime', standard_models = {
            'low_extinction': 'atm_trans_2156_1_3_2_0_0_0.1_0.1.dat',
            'navy_maritime': 'atm_trans_2156_1_3_0_0_0.dat' },
        quiet = False):
    model_path = standard_models[absorption_model] if absorption_model in standard_models else absorption_model
    atm_abs = calin.simulation.detector_efficiency.AtmosphericAbsorption(ds_filename(model_path))

    if not quiet:
        print('Loading atmospheric absorption model :', model_path)

    return atm_abs

def default_optical_model_datapack_filename():
    return 'panoseti_optical_model_data_pack.json'

def read_optical_model_datapack(filename = None):
    if filename is None:
        filename = default_optical_model_datapack_filename()
    resolved_filename = ds_filename(filename)
    with open(resolved_filename, 'r') as f:
        file_record = calin.provenance.chronicle.register_file_open(
            resolved_filename,
            calin.ix.provenance.chronicle.AT_READ,
            'calin.simulation.vs_panoseti.read_optical_model_datapack')
        datapack = json.load(f)
        calin.provenance.chronicle.register_file_close(file_record)
    return datapack

def detection_efficiency_from_datapack(datapack = None, degradation_factor = 1.0, quiet = False):
    if datapack is None:
        datapack = read_optical_model_datapack()

    det_eff = calin.simulation.detector_efficiency.DetectionEfficiency()

    # SiPM PDE curve
    if 'sipm_pde' in datapack:
        sipm_pde = datapack['sipm_pde']
        sipm_eff = calin.simulation.detector_efficiency.DetectionEfficiency()
        for ev, eff in zip(sipm_pde['ev'], sipm_pde['eff']):
            sipm_eff.insert(ev, eff)
        det_eff.scaleEff(sipm_eff)
        if not quiet:
            print('Scaled by SiPM PDE [bandwidth = %.3f]' % det_eff.integrate())

    # PMMA transmission curve
    if 'pmma_transmission' in datapack:
        pmma = datapack['pmma_transmission']
        pmma_eff = calin.simulation.detector_efficiency.DetectionEfficiency()
        for ev, eff in zip(pmma['ev'], pmma['eff']):
            pmma_eff.insert(ev, eff)
        det_eff.scaleEff(pmma_eff)
        if not quiet:
            print('Scaled by PMMA transmission [bandwidth = %.3f]' % det_eff.integrate())

    # Degradation factor
    if degradation_factor < 1.0:
        det_eff.scaleEffByConst(degradation_factor)
        if not quiet:
            print('Scaled by degradation factor : %.3f [bandwidth = %.3f]' % (
                degradation_factor, det_eff.integrate()))

    return det_eff

def lens_refractive_index_spline_from_datapack(datapack = None):
    if datapack is None:
        datapack = read_optical_model_datapack()

    ref_data = datapack['refractive_index']
    ev = list(ref_data['ev'])
    n = list(ref_data['n'])

    # CubicSpline requires knots sorted ascendingly in x
    sorted_pairs = sorted(zip(ev, n), key=lambda t: t[0])
    ev_sorted = [p[0] for p in sorted_pairs]
    n_sorted = [p[1] for p in sorted_pairs]

    return calin.math.spline_interpolation.CubicSpline(ev_sorted, n_sorted)

def dark100_impulse_response(pulse_file = 'Pulse_template_dark100.dat',
                            pulse_length_ns = 60.0, pulse_decay_ns = 3.5,
                            sample_period_ns = 1.0):
    """Return the Dark100 waveform impulse response as ``hg``, ``lg`` and ``dt``.

    Until the digitized response file is installed, return an all-zero response.
    Add ``Pulse_template_dark100.dat`` to the simulation data directory to load
    the digitized curve. The file format is columns ``time_ns, hg`` or
    ``time_ns, hg, lg``; a two-column curve is used for both gains.
    """
    resolved_file = ds_filename(pulse_file)
    if not os.path.exists(resolved_file):
        nsample = int(numpy.ceil(pulse_length_ns / sample_period_ns))
        return dict(hg=numpy.zeros(nsample), lg=numpy.zeros(nsample),
                    dt=sample_period_ns)

    with open(resolved_file, 'r') as pulse_stream:
        file_record = calin.provenance.chronicle.register_file_open(
            resolved_file, calin.ix.provenance.chronicle.AT_READ,
            'calin.simulation.vs_panoseti.dark100_impulse_response')
        comments = ''.join(line for line in pulse_stream if line.startswith('#'))
        file_record.set_comment(comments)
        calin.provenance.chronicle.register_file_close(file_record)

    pulse = numpy.loadtxt(resolved_file, comments='#', ndmin=2)
    if pulse.shape[1] not in (2, 3):
        raise ValueError('Dark100 pulse file must have 2 or 3 columns: time, hg[, lg]')
    t = pulse[:, 0]
    dt = float(numpy.mean(numpy.diff(t))) if len(t) > 1 else sample_period_ns
    nsample = int(numpy.ceil(pulse_length_ns / dt))
    if nsample < len(pulse):
        raise ValueError('Dark100 pulse file is longer than pulse_length_ns')

    def pad_gain(gain):
        response = numpy.zeros(nsample)
        response[:len(pulse)] = pulse[:, gain]
        if nsample > len(pulse):
            n = nsample - len(pulse)
            decay = 0.5 * (1.0 - numpy.tanh(
                (numpy.arange(n) - n / 2) / pulse_decay_ns * dt))
            response[len(pulse):] = pulse[-1, gain] * decay
        return response

    return dict(hg=pad_gain(1),
                lg=pad_gain(2) if pulse.shape[1] == 3 else pad_gain(1),
                dt=dt)

def array_parameters_from_datapack(datapack = None, elevation = 60,
                                   scope_x = 0, scope_y = 0, scope_z = 0,
                                   array_lat = None, array_lon = None, array_alt = None):
    if datapack is None:
        datapack = read_optical_model_datapack()

    opt_model = datapack['optical_model']
    params = calin.ix.simulation.panoseti_optics.ArrayParameters()

    lat_default, lon_default, alt_default = palomar_location()
    lat = lat_default if array_lat is None else array_lat
    lon = lon_default if array_lon is None else array_lon
    alt = alt_default if array_alt is None else array_alt

    params.mutable_array_origin().set_latitude(lat)
    params.mutable_array_origin().set_longitude(lon)
    params.mutable_array_origin().set_elevation(alt)

    # Set telescope positions
    try:
        npos = max(len(scope_x), len(scope_y))
        sx = list(scope_x) if isinstance(scope_x, (list, tuple, numpy.ndarray)) else [scope_x] * npos
        sy = list(scope_y) if isinstance(scope_y, (list, tuple, numpy.ndarray)) else [scope_y] * npos
        sz = list(scope_z) if isinstance(scope_z, (list, tuple, numpy.ndarray)) else [scope_z] * npos
        for x, y, z in zip(sx, sy, sz):
            pos = params.add_scope_positions()
            pos.set_x(x)
            pos.set_y(y)
            pos.set_z(z)
    except Exception:
        pos = params.add_scope_positions()
        pos.set_x(scope_x)
        pos.set_y(scope_y)
        pos.set_z(scope_z)

    # Populate lens and detector optics parameters
    params.set_fresnel_lens_aperture(float(opt_model['D']))
    params.set_fresnel_lens_thickness(float(opt_model['thickness']))
    params.set_fresnel_lens_groove_width(float(opt_model['groove_width']))
    params.set_fresnel_lens_draft_angle(float(opt_model['draft_angle']))
    params.set_fresnel_lens_roughness(float(opt_model.get('roughness', 0.0)))
    params.set_detector_separation(float(opt_model['F']))
    params.set_pixel_pitch(float(opt_model['pixel_spacing']))
    params.set_num_pixels_per_axis(int(opt_model['npixel']))

    for c in opt_model['p_out']:
        params.add_fresnel_lens_polynomial(float(c))

    return params

def dark100_palomar_config(elevation = 60, datapack = None):
    # TELESCOPE    -0.10E2   97.18E2    39.70E2		25	# Heli
    # TELESCOPE	 -220.15E2   33.67E2	49.17E2		25	# Winter
    # TELESCOPE  -130.43E2	190.40E2	34.66E2		25	# Fern
    # TELESCOPE   177.58E2 -333.33E2	42.82E2		25	# Gattini
    # TELESCOPE  -402.48E2  133.47E2	45.23E2		25	# Tower
    # TELESCOPE   132.97E23  -9.01E2	37.22E2		25	# Antler
    # TELESCOPE     0.10E2  -97.18E2	46.35E2		25	# Vent

    scope_pos = [
        [   -0.10E2,   97.18E2, 39.70E2 ],  # Heli
        [ -220.15E2,   33.67E2, 49.17E2 ],  # Winter
        [ -130.43E2,  190.40E2, 34.66E2 ],  # Fern
        [  177.58E2, -333.33E2, 42.82E2 ],  # Gattini
        [ -402.48E2,  133.47E2, 45.23E2 ],  # Tower
        [  132.97E2,   -9.01E2, 37.22E2 ],  # Antler
        [    0.10E2,  -97.18E2, 46.35E2 ]   # Vent
    ]

    scope_x = [p[0] for p in scope_pos]
    scope_y = [p[1] for p in scope_pos]
    scope_z = [p[2] for p in scope_pos]
    return array_parameters_from_datapack(
        datapack = datapack, elevation = elevation,
        scope_x = scope_x, scope_y = scope_y, scope_z = scope_z)

/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

// General includes
#include "GRParmParse.hpp"
#include "SimulationParametersBase.hpp"

// Problem specific includes:
#include "InitialData.hpp"
#include "KerrBH.hpp"
#include "Potential.hpp"

class SimulationParameters : public SimulationParametersBase
{
  public:
    SimulationParameters(GRParmParse &pp) : SimulationParametersBase(pp)
    {
        // read the problem specific params
        read_params(pp);
        check_params();
    }

    void read_params(GRParmParse &pp)
    {
        // Initial scalar field data
        pp.load("G_Newton", G_Newton);
        pp.load("scalar_amplitude", initial_params.amplitude, 0.1);
        pp.load("N_efolds", initial_params.N_efolds, 1.0);
        pp.load("scalar_mass", potential_params.scalar_mass, 0.1);
        pp.load("threshold_K", threshold_K);
        pp.load("threshold_phi", threshold_phi);

        
        pp.load("calculate_constraint_norms", calculate_constraint_norms, false);
    }

    void check_params()
    {
    }

    // Initial data for matter and potential and BH
    double G_Newton;
    bool calculate_constraint_norms;
    double threshold_K;
    double threshold_phi;
    InitialData::params_t initial_params;
    Potential::params_t potential_params;
    KerrBH::params_t kerr_params;
};

#endif /* SIMULATIONPARAMETERS_HPP_ */

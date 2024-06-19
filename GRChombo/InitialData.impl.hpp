/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(INITIALDATA_HPP_)
#error "This file should only be included through InitialData.hpp"
#endif

#ifndef INITIALDATA_IMPL_HPP_
#define INITIALDATA_IMPL_HPP_

inline InitialData::InitialData(params_t a_params, double a_dx)
    : m_dx(a_dx), m_params(a_params)
{
}

// Compute the value of the initial vars on the grid
template <class data_t>
void InitialData::compute(Cell<data_t> current_cell) const
{
    MatterCCZ4RHS<CosmScalarField<>>::Vars<data_t> vars;
    VarsTools::assign(vars, 0.); // Set only the non-zero components below

    // set the field vars (DBaumann)
    vars.phi = 2*sqrt((m_params.N_efolds-0.5)/(8*M_PI));
    vars.Pi = -sqrt(6/(8*M_PI))*m_params.scalar_mass;

    // start with unit lapse and flat metric (must be relaxed for chi)
    vars.lapse = 1.0;
    vars.chi = 1.0;

    // conformal metric is flat
    FOR(i) vars.h[i][i] = 1.;
    
    vars.K = -sqrt(24.0*M_PI*(pow(vars.Pi, 2)/2.0+pow(vars.phi, 2)/2.0)); //katyClough
    
    // Store the initial values of the variables
    current_cell.store_vars(vars);
}

#endif /* InitialData_IMPL_HPP_ */

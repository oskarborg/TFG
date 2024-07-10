/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "ScalarFieldLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SixthOrderDerivatives.hpp"
#include "TraceARemoval.hpp"

// For RHS update
#include "MatterCCZ4RHS.hpp"

// For constraints calculation
#include "NewMatterConstraints.hpp"

// For tag cells
#include "PhiAndKTaggingCriterion.hpp"

// Problem specific includes
#include "ComputePack.hpp"
#include "GammaCalculator.hpp"
#include "InitialData.hpp"
#include "KerrBH.hpp"
#include "Potential.hpp"
#include "CosmScalarField.hpp"
#include "SetValue.hpp"
#include "SmallDataIO.hpp"
#include "AMRReductions.hpp"
//#include "ComputeEps.hpp"
//#include "ComputeVariance.hpp"
//#include "SetAvgK.hpp"

// Things to do at each advance step, after the RK4 is calculated
void ScalarFieldLevel::specificAdvance()
{
    fillAllGhosts();
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(
        make_compute_pack(TraceARemoval(),
                          PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(), m_state_new, m_state_new,
                       EXCLUDE_GHOST_CELLS, disable_simd());
    //BoxLoops::loop(ComputeEps(m_p.potential_params), m_state_new, m_state_new, EXCLUDE_GHOST_CELLS);
}

// Initial data for field and metric variables
void ScalarFieldLevel::initialData()
{
    CH_TIME("ScalarFieldLevel::initialData");
    if (m_verbosity)
        pout() << "ScalarFieldLevel::initialData " << m_level << endl;

    // First set everything to zero then initial conditions for scalar field -
    // here a Kerr BH and a scalar field profile
    BoxLoops::loop(
        make_compute_pack(SetValue(0.), InitialData(m_p.initial_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(
        	MatterConstraints<ScalarFieldWithPotential>(
       		scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom)),
      		m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
    double L2_Ham = amr_reductions.norm(c_Ham);
    pout() << "Ham. Constraint al inicio: " << L2_Ham << "\n";

}

#ifdef CH_USE_HDF5
// Things to do before outputting a checkpoint file
void ScalarFieldLevel::prePlotLevel()
{
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(
        MatterConstraints<ScalarFieldWithPotential>(
            scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom)),
        m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}
#endif

// Things to do in RHS update, at each RK4 step
void ScalarFieldLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                       const double a_time)
{
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(
        make_compute_pack(TraceARemoval(),
                          PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
        a_soln, a_soln, INCLUDE_GHOST_CELLS);
        
    /*//get the avg of K (for the gauge update)
    AMRReductions<VariableType::evolution> amr_reductions(m_gr_amr);
    double avgK = amr_reductions.sum(c_K)/amr_reductions.get_domain_volume();
    BoxLoops::loop(SetAvgK(avgK), m_state_new, m_state_new, EXCLUDE_GHOST_CELLS);*/

    // Calculate MatterCCZ4 right hand side with matter_t = ScalarField
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    if (m_p.max_spatial_derivative_order == 4)
    {
        MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
                      FourthOrderDerivatives>
            my_ccz4_matter(scalar_field, m_p.ccz4_params, m_dx, m_p.sigma,
                           m_p.formulation, m_p.G_Newton);
        BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
    else if (m_p.max_spatial_derivative_order == 6)
    {
        MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
                      SixthOrderDerivatives>
            my_ccz4_matter(scalar_field, m_p.ccz4_params, m_dx, m_p.sigma,
                           m_p.formulation, m_p.G_Newton);
        BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
}

// Things to do at ODE update, after soln + rhs
void ScalarFieldLevel::specificUpdateODE(GRLevelData &a_soln,
                                         const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce trace free A_ij
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void ScalarFieldLevel::preTagCells()
{
    // We only use chi in the tagging criterion so only fill the ghosts for chi
    fillAllGhosts();
}

void ScalarFieldLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                               const FArrayBox &current_state)
{
    //BoxLoops::loop(PhiAndKTaggingCriterion(m_dx, m_p.threshold_phi, m_p.threshold_K),
     //                  current_state, tagging_criterion);
}


void ScalarFieldLevel::specificPostTimeStep()
{
    CH_TIME("BinaryBHLevel::specificPostTimeStep");

    bool first_step =
        (m_time == m_dt); // this form is used when 'specificPostTimeStep' was
                        // called during setup at t=0 from Main
    // bool first_step = (m_time == m_dt); // if not called in Main
    
    if (m_p.calculate_constraint_norms)
    {
        fillAllGhosts();
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
 	BoxLoops::loop(
        	MatterConstraints<ScalarFieldWithPotential>(
       		scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom)),
      		m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
        if (m_level == 0)
        {
            AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
            double L2_Ham = amr_reductions.norm(c_Ham);
            double L2_Mom = amr_reductions.norm(Interval(c_Mom, c_Mom));
            SmallDataIO constraints_file(m_p.data_path + "constraint_norms",
                                         m_dt, m_time, m_restart_time,
                                         SmallDataIO::APPEND, first_step);
            constraints_file.remove_duplicate_time_data();
            if (first_step)
            {
                constraints_file.write_header_line({"L^2_Ham", "L^2_Mom"});
            }
            constraints_file.write_time_data_line({L2_Ham, L2_Mom});
        }
    }
    
    //Output the number of efolds and the epsilon since the beggining of the simulation
    /*
    	fillAllGhosts();
        AMRReductions<VariableType::evolution> amr_reductions(m_gr_amr);
    if(m_level==0)
    {
        double Nefolds = amr_reductions.sum(c_chi)/amr_reductions.get_domain_volume();
        Nefolds=-log(Nefolds);
        double eps = amr_reductions.sum(c_epsilon)/amr_reductions.get_domain_volume(); 
        double phi = amr_reductions.sum(c_phi)/amr_reductions.get_domain_volume(); 
        BoxLoops::loop(ComputeVariance(phi), m_state_new, m_state_new, EXCLUDE_GHOST_CELLS);
        double Pi = amr_reductions.sum(c_Pi)/amr_reductions.get_domain_volume();
        double K = amr_reductions.sum(c_K)/amr_reductions.get_domain_volume();
        double variance = amr_reductions.sum(c_variance)/amr_reductions.get_domain_volume(); 
        SmallDataIO Nefolds_file(m_p.data_path + "Inflation",
                                         m_dt, m_time, m_restart_time,
                                         SmallDataIO::APPEND, first_step);
            Nefolds_file.remove_duplicate_time_data();
            if (first_step)
            {
               Nefolds_file.write_header_line({"Nefolds", "epsilon", "phi", "Pi", "K", "Variance"});
            }
            Nefolds_file.write_time_data_line({Nefolds, eps, phi, Pi, K, variance});
    }*/
}

#ifndef _SETLEVELDATA_H_
#define _SETLEVELDATA_H_

#include <iostream>
using std::cerr;

#include "AMRMultiGrid.H"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "GRChomboUserVariables.hpp"
#include "MultigridUserVariables.hpp"
#include "ParmParse.H"
#include "PoissonParameters.H"
#include "VariableCoeffPoissonOperator.H"
#include "VariableCoeffPoissonOperatorFactory.H"

// various functions operating on grids to set up data for initial
// conditions, spatially dependent coefficients and rhs etc

inline void get_loc(RealVect &a_out_loc, const IntVect &a_iv,
                    const RealVect &a_dx, const PoissonParameters &a_params);


void set_initial_conditions(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_dW,
                            const RealVect &a_dx, const PoissonParameters &a_params, Vector<Vector<Vector<Vector<double>>>> rnd_phases);

                                   
inline Real get_grad_sq(const IntVect &a_iv, const FArrayBox &a_phi_fab,
                            const RealVect &a_dx);
                                  
                                  
extern void set_regrid_condition(LevelData<FArrayBox> &a_condition,
                                 LevelData<FArrayBox> &a_multigrid_vars,
                                 const RealVect &a_dx,
                                 const PoissonParameters &a_params);


extern void set_b_coef(LevelData<FArrayBox> &a_bCoef,
                       const PoissonParameters &a_params, const RealVect &a_dx);


extern void set_a_coef(LevelData<FArrayBox> &a_bCoef,
                       const PoissonParameters &a_params, const RealVect &a_dx);

extern void set_K(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_rhs,
                       const PoissonParameters &a_params, const RealVect &a_dx, const Copier &a_exchange_copier);
                       

void set_rhs(LevelData<FArrayBox> &a_rhs,
             LevelData<FArrayBox> &a_multigrid_vars, const RealVect &a_dx,
             const PoissonParameters &a_params, const Copier &a_exchange_copier);
             
void norm_dW(LevelData<FArrayBox> &a_dW, RealVect dWNorm);
void norm_W(LevelData<FArrayBox> &a_dW, RealVect dWNorm);
                       
                       
void set_output_data(LevelData<FArrayBox> &a_grchombo_vars,
                     LevelData<FArrayBox> &a_multigrid_vars,
                     const PoissonParameters &a_params, const RealVect &a_dx);
                     
                     
void set_update_W_i(LevelData<FArrayBox> &a_multigrid_vars,
                     LevelData<FArrayBox> &a_dW,
                     const Copier &a_exchange_copier);
                     
void set_A_ij(LevelData<FArrayBox> &a_multigrid_vars,LevelData<FArrayBox> &a_rhs,
                              const RealVect &a_dx, const PoissonParameters &a_params, const Copier &a_exchange_copier);
                              
void get_Const(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_ConstLevel, 
                     const PoissonParameters &a_params, const RealVect &a_dx);
#endif

#include "SetLevelData.H"
#include "AMRIO.H"
#include "BCFunc.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "LoadBalance.H"
#include "MyPhiFunction.H"
#include "PoissonParameters.H"
#include "SetBinaryBH.H"
#include "VariableCoeffPoissonOperatorFactory.H"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>

// Set various LevelData functions across the grid

// This takes an IntVect and writes the physical coordinates to a RealVect
inline void get_loc(RealVect &a_out_loc, const IntVect &a_iv,
                    const RealVect &a_dx, const PoissonParameters &a_params)
{
    a_out_loc = a_iv + 0.5 * RealVect::Unit;
    a_out_loc *= a_dx;
    a_out_loc -= a_params.domainLength / 2.0;
}


// computes the Laplacian of W_i at a point in a box
inline Real get_laplacian_W_i(const IntVect &a_iv, const FArrayBox &a_W_i_fab,
                              const RealVect &a_dx)
{    Real laplacian_of_W_i = 0.0;
    for (int i = -1; i <= 1; ++i)
    {
        for(int j = -1; j <= 1; j++)
        {
            for(int k = -1; k<=1; k++)
            {
                IntVect iv_offset1 = a_iv;
                iv_offset1[0] += i;
                iv_offset1[1] += j;
                iv_offset1[2] += k;

                // 2nd order stencil for now
                Real coef;
                int a = abs(i)+abs(j)+abs(k);
                if(a==3){
                    coef=0;
                }else if(a==2){
                    coef=0;
                }else if(a==1){
                    coef=1;
                }else{
                    coef=-6;
                }
                Real d2W_i_dxdx = a_W_i_fab(iv_offset1);   
                laplacian_of_W_i += coef*d2W_i_dxdx;
            }
        }
    }
    return laplacian_of_W_i/(a_dx[0]*a_dx[0]);
} // end get_laplacian_W_i


// S_i = -Pi Der_i phi
inline Real get_S_i(const FArrayBox &a_phi_fab, const Real &Pi_here,
            int idir, const IntVect &a_iv, const RealVect &a_dx)
{
    //Calculate the derivative of phi
    IntVect iv_offset1 = a_iv;
    IntVect iv_offset2 = a_iv;
    iv_offset1[idir] -= 1;
    iv_offset2[idir] += 1;
    
    Real dphi_i = 0.5 * (a_phi_fab(iv_offset2) - a_phi_fab(iv_offset1)) / a_dx[idir]; //2nd order stencils

    Real S_i=-Pi_here*dphi_i;

    return S_i;
}


// set initial guess value for the conformal factor psi
// defined by \gamma_ij = \psi^4 \tilde \gamma_ij, scalar field phi
// and \bar Aij = psi^2 A_ij.
// For now the default setup is 2 Bowen York BHs plus a scalar field
// with some initial user specified configuration
void set_initial_conditions(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_dW,
                            const RealVect &a_dx, const PoissonParameters &a_params, Vector<Vector<Vector<Vector<double>>>> rnd_phases)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);
    
    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dW_box = a_dW[dit()];
        
        Box b = multigrid_vars_box.box();
        BoxIterator bit(b);
        for (bit.begin(); bit.ok(); ++bit)
        {

            // work out location on the grid
            IntVect iv = bit();

            // set psi to 1.0 and zero dpsi
            // note that we don't include the singular part of psi
            // for the BHs - this is added at the output data stage
            // and when we calculate psi_0 in the rhs etc
            // as it already satisfies Laplacian(psi) = 0
            multigrid_vars_box(iv, c_psi_reg) = 1.0;
            for(int i = 0; i<=2; i++)
            {
                dW_box(iv, i) = 0.0;
            }

            // set the phi value - need the distance from centre
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);
            
            // set phi according to user defined function
            multigrid_vars_box(iv, c_phi_0) =
                my_phi_function(loc, a_params.phi_amplitude,
                                a_params.nCells, a_params.domainLength, a_params.phi_0, rnd_phases);
            multigrid_vars_box(iv, c_Pi_0) =
                my_Pi_function(loc, a_params.phi_amplitude,
                                a_params.nCells, a_params.domainLength, a_params.phi_0, a_params.Pi_0, multigrid_vars_box(iv, c_phi_0), rnd_phases);
            
            
            multigrid_vars_box(iv, c_A11_0) = 0.0;
            multigrid_vars_box(iv, c_A12_0) = 0.0;
            multigrid_vars_box(iv, c_A13_0) = 0.0;
            multigrid_vars_box(iv, c_A22_0) = 0.0;
            multigrid_vars_box(iv, c_A23_0) = 0.0;
            multigrid_vars_box(iv, c_A33_0) = 0.0;
            
        }
    }
    pout() << "Initial conditions set";
} // end set_initial_conditions


// computes the gradient of the scalar field squared at a point in a box
// i.e. \delta^{ij} d_i phi d_j phi
inline Real get_grad_sq(const IntVect &a_iv, const FArrayBox &a_phi_fab,
                            const RealVect &a_dx)
{	
    Real grad_phi_sq = 0.0;
    for (int idir = 0; idir < SpaceDim; ++idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;
  
        // 2nd order stencils for now
        Real dphi_dx =
            0.5 * (a_phi_fab(iv_offset2) - a_phi_fab(iv_offset1)) / a_dx[idir];

        grad_phi_sq += dphi_dx * dphi_dx;
    }
    return 0;//grad_phi_sq;
} // end get_grad_phi_sq

// set the regrid condition - abs value of this drives AMR regrid
void set_regrid_condition(LevelData<FArrayBox> &a_condition,
                          LevelData<FArrayBox> &a_multigrid_vars,
                          const RealVect &a_dx,
                          const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_condition.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &condition_box = a_condition[dit()];
        condition_box.setVal(0.0, 0);
        Box this_box = condition_box.box(); // no ghost cells

        
        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);
        FArrayBox K_fab(Interval(c_K_0, c_K_0), multigrid_vars_box);
        condition_box.setVal(0.0, 0);
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);
    
            // condition = dx * (grad_K / K_Threshold + grad_phi / phi_Threshold)
            Real grad_K_sq = get_grad_sq(iv, K_fab, a_dx);
            Real grad_phi_sq = get_grad_sq(iv, phi_fab, a_dx);
            Real dx = a_dx[0]*a_dx[1]*a_dx[2];
    
            condition_box(iv, 0) = dx * (sqrt(grad_K_sq) / a_params.K_Cond + sqrt(grad_phi_sq) / a_params.phi_Cond);
        }
    }
} // end set_regrid_condition

// rho = 0.5 Pi^2+0.5*grad_phi^2+V(phi)
inline Real get_rho(const Real &phi_here, const Real &Pi_here, 
            const Real &grad_phi_sq_here, const PoissonParameters &a_params)
{
    Real V_of_phi = 0.25*pow(phi_here, 4)*pow(10, -4);  //TODO: mass term
    //
    Real rho = 0.5 * Pi_here * Pi_here + V_of_phi + 0.5*grad_phi_sq_here;

    return rho;
}

// Derivative of W_j
inline Real D_i_W_j(const FArrayBox &a_W_fab, int i, int j,
                        const IntVect &a_iv, const RealVect &a_dx)
{
    //Calculate the derivative of W_j
    IntVect iv_offset1 = a_iv;
    IntVect iv_offset2 = a_iv;
    iv_offset1[i] -= 1;
    iv_offset2[i] += 1;
    Real D_i_W_j = 0.5 * (a_W_fab(iv_offset2, j) - a_W_fab(iv_offset1, j)) / a_dx[i]; //2nd order stencils

    return D_i_W_j;
}

// Derivative of K
inline Real get_d_i_K(const FArrayBox &a_K_fab, int i,
                        const IntVect &a_iv, const RealVect &a_dx)
{
    //Calculate the derivative of W_j
    IntVect iv_offset1 = a_iv;
    IntVect iv_offset2 = a_iv;
    iv_offset1[i] -= 1;
    iv_offset2[i] += 1;
    Real d_i_K = 0.5 * (a_K_fab(iv_offset2, 0) - a_K_fab(iv_offset1, 0)) / a_dx[i]; //2nd order stencils

    return d_i_K;
}


// The coefficient of the Laplacian operator, for now set to constant 1
// Note that beta = -1 so this sets the sign
// the rhs source of the Poisson eqn
void set_b_coef(LevelData<FArrayBox> &a_bCoef,
                const PoissonParameters &a_params, const RealVect &a_dx)
{

    CH_assert(a_bCoef.nComp() == 1);
    int comp_number = 0;

    for (DataIterator dit = a_bCoef.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &bCoef_box = a_bCoef[dit()];
        bCoef_box.setVal(1.0, comp_number);
    }
}

// For now a_coef=0
void set_a_coef(LevelData<FArrayBox> &a_bCoef,
                const PoissonParameters &a_params, const RealVect &a_dx)
{

    CH_assert(a_bCoef.nComp() == 1);
    int comp_number = 0;

    for (DataIterator dit = a_bCoef.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &bCoef_box = a_bCoef[dit()];
        bCoef_box.setVal(1.0, comp_number);
    }
}


//algebraically determine K
extern void set_K(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_rhs,
                       const PoissonParameters &a_params, const RealVect &a_dx, const Copier &a_exchange_copier)
{
    a_multigrid_vars.exchange(a_multigrid_vars.interval(), a_exchange_copier);
    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        Box this_box = multigrid_vars_box.box();

        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);
            
            Real K2;
            // K^2=3/2 A_ij A^ij+24pi rho
            Real grad_phi_sq = get_grad_sq(iv, phi_fab, a_dx);
            Real rho = get_rho(multigrid_vars_box(iv, c_phi_0), multigrid_vars_box(iv, c_Pi_0),
                    grad_phi_sq, a_params);
            

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
            pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                    pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                    2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                    2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                    2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            K2=1.5*A2+24*M_PI*rho*a_params.G_Newton;

            multigrid_vars_box(iv, c_K_0)=-sqrt(K2);
        }
    }
}


// set the rhs source for the poisson eqn
void set_rhs(LevelData<FArrayBox> &a_rhs,
             LevelData<FArrayBox> &a_multigrid_vars, const RealVect &a_dx,
             const PoissonParameters &a_params, const Copier &a_exchange_copier)
{

    a_multigrid_vars.exchange(a_multigrid_vars.interval(), a_exchange_copier);
    
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);
    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        Box this_box = rhs_box.box(); // no ghost cells
        
        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);
        FArrayBox K_fab(Interval(c_K_0, c_K_0), multigrid_vars_box);
        FArrayBox W1_fab(Interval(c_W1, c_W1), multigrid_vars_box);
        FArrayBox W2_fab(Interval(c_W2, c_W2), multigrid_vars_box);
        FArrayBox W3_fab(Interval(c_W3, c_W3), multigrid_vars_box);
        
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
        
            IntVect iv = bit();
           
            RealVect rhs_i(0.0 , 0.0 ,0.0);
            RealVect laplW_i(0.0, 0.0, 0.0);
            
            laplW_i[0] = get_laplacian_W_i(iv, W1_fab, a_dx);
            laplW_i[1] = get_laplacian_W_i(iv, W2_fab, a_dx);
            laplW_i[2] = get_laplacian_W_i(iv, W3_fab, a_dx);
            
            for(int i=0; i<=2; i++) //Each component
            {
                
                Real d_i_K = get_d_i_K(K_fab, i, iv, a_dx);
                
                Real S_i = get_S_i(phi_fab, multigrid_vars_box(iv, c_Pi_0), i, iv, a_dx);
                                   
                // rhs = 0.5 K_grad + 6 pi S_i - laplacian(W_i)
                rhs_i[i] = 0.5*d_i_K+6*M_PI*S_i*a_params.G_Newton-laplW_i[i];
            }
            for(int i=0; i<=2; i++)
            {
                rhs_box(iv, i) = rhs_i[i];
            }
        }
    }
} // end set_rhs


void norm_dW(LevelData<FArrayBox> &a_dW, RealVect dWNorm)
{
    DataIterator dit = a_dW.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &dW_box = a_dW[dit()];

        Box this_box = dW_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            for(int i=0;i<=2;i++)
            {
                dW_box(iv, i) -= dWNorm[i];
            }
        }
    }
}

void norm_W(LevelData<FArrayBox> &a_dW, RealVect dWNorm)
{
    DataIterator dit = a_dW.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &dW_box = a_dW[dit()];

        Box this_box = dW_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            for(int i=0;i<=2;i++)
            {
                dW_box(iv, 2+i) -= dWNorm[i];
            }
        }
    }
}

// Add the correction to W_i after the solver operates
void set_update_W_i(LevelData<FArrayBox> &a_multigrid_vars,
                     LevelData<FArrayBox> &a_dW,
                     const Copier &a_exchange_copier)
{

    // first exchange ghost cells for dpsi so they are filled with the correct
    // values
    a_dW.exchange(a_dW.interval(), a_exchange_copier);

    DataIterator dit = a_dW.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dW_box = a_dW[dit()];

        Box this_box = dW_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            for(int i=0;i<=2;i++)
            {
                multigrid_vars_box(iv, 2+i) += dW_box(iv, i);
            }
        }
    }
    
    
}


//Calculates new A_ij
void set_A_ij(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_rhs,
                              const RealVect &a_dx, const PoissonParameters &a_params, const Copier &a_exchange_copier)
{
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);
        // first exchange ghost cells for dpsi so they are filled with the correct
    // values
    a_multigrid_vars.exchange(a_multigrid_vars.interval(), a_exchange_copier);

    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        FArrayBox W_fab(Interval(c_W1, c_W3), multigrid_vars_box);
        Box this_box = rhs_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);
            
            Real div_W = 0.0;
            for(int i=0; i<=2; i++)
            {
                div_W += D_i_W_j(W_fab, i, i, iv, a_dx);
            }
            

            //A_ij=D_i W_j + D_j W_i - 1.5 g_ij D_k W^k
            for(int i = 0; i <=2; i++)
            {
                for(int j=i; j <= 2; j++)
                {
                
                    Real Aij = D_i_W_j(W_fab, i, j, iv, a_dx) + D_i_W_j(W_fab, j, i, iv, a_dx);
                    if(i==j)
                    {
                        Aij -= 1.5 * div_W;
                    }
                    multigrid_vars_box(iv, 5 +(5-i)*i/2+j) = Aij;
                }
            }
        }
    }
}

// used to set output data for all ADM Vars for GRChombo restart
void set_output_data(LevelData<FArrayBox> &a_grchombo_vars,
                     LevelData<FArrayBox> &a_multigrid_vars,
                     const PoissonParameters &a_params, const RealVect &a_dx)
{

    CH_assert(a_grchombo_vars.nComp() == NUM_GRCHOMBO_VARS);
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_grchombo_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &grchombo_vars_box = a_grchombo_vars[dit()];
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];

        // first set everything to zero
        for (int comp = 0; comp < NUM_GRCHOMBO_VARS; comp++)
        {
            grchombo_vars_box.setVal(0.0, comp);
        }

        // now set non zero terms - const across whole box
        // Conformally flat, and lapse = 1
        grchombo_vars_box.setVal(1.0, c_h11);
        grchombo_vars_box.setVal(1.0, c_h22);
        grchombo_vars_box.setVal(1.0, c_h33);
        grchombo_vars_box.setVal(1.0, c_lapse);


        // now non constant terms by location
        Box this_box = grchombo_vars_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // GRChombo conformal factor chi = psi^-4
            Real chi = pow(multigrid_vars_box(iv, c_psi_reg), -4.0);
            grchombo_vars_box(iv, c_chi) = chi;
            Real factor = pow(chi, 1.5);
            
            //K
            grchombo_vars_box(iv, c_K) = multigrid_vars_box(iv, c_K_0);

            // Copy phi and Aij across - note this is now \tilde Aij not \bar
            // Aij
            grchombo_vars_box(iv, c_phi) = multigrid_vars_box(iv, c_phi_0);
            grchombo_vars_box(iv, c_Pi) = multigrid_vars_box(iv, c_Pi_0);
            grchombo_vars_box(iv, c_A11) =
                multigrid_vars_box(iv, c_A11_0) * factor;
            grchombo_vars_box(iv, c_A12) =
                multigrid_vars_box(iv, c_A12_0) * factor;
            grchombo_vars_box(iv, c_A13) =
                multigrid_vars_box(iv, c_A13_0) * factor;
            grchombo_vars_box(iv, c_A22) =
                multigrid_vars_box(iv, c_A22_0) * factor;
            grchombo_vars_box(iv, c_A23) =
                multigrid_vars_box(iv, c_A23_0) * factor;
            grchombo_vars_box(iv, c_A33) =
                multigrid_vars_box(iv, c_A33_0) * factor;
        }
    }
}


void get_Const(LevelData<FArrayBox> &a_multigrid_vars, LevelData<FArrayBox> &a_ConstLevel, 
                     const PoissonParameters &a_params, const RealVect &a_dx)
{
    DataIterator dit = a_ConstLevel.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &const_vars_box = a_ConstLevel[dit()];
        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);
        Box this_box = const_vars_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
            pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                    pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                    2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                    2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                    2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);
            Real grad_phi_sq = get_grad_sq(iv, phi_fab, a_dx);
            Real rho = get_rho(multigrid_vars_box(iv, c_phi_0), multigrid_vars_box(iv, c_Pi_0),
                    grad_phi_sq, a_params);
            const_vars_box(iv, 3) = pow(multigrid_vars_box(iv, c_K_0)*multigrid_vars_box(iv, c_K_0)
                    -1.5*A2-24*M_PI*rho , 2);
        }
    }
    
    //The mom. constraints are eq to rhs
    //set_rhs(a_ConstLevel,a_multigrid_vars, a_dx, a_params);
}


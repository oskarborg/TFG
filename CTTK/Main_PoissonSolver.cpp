/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "DebugDump.H"
#include "FABView.H"
#include "FArrayBox.H"
#include "LevelData.H"
#include "LoadBalance.H"
#include "MultilevelLinearOp.H"
#include "ParmParse.H"
#include "PoissonParameters.H"
#include "SetBCs.H"
#include "SetGrids.H"
#include "SetLevelData.H"
#include "UsingNamespace.H"
#include "VariableCoeffPoissonOperatorFactory.H"
#include "WriteOutput.H"
#include "computeNorm.H"
#include "computeSum.H"
#include <iostream>
#include <random>

#ifdef CH_Linux
// Should be undefined by default
//#define TRAP_FPE
#undef TRAP_FPE
#endif

#ifdef TRAP_FPE
static void enableFpExceptions();
#endif

using std::cerr;

// Sets up and runs the solver
// The equation solved is: [aCoef*I + bCoef*Laplacian](dpsi) = rhs
// We assume conformal flatness, K=const and Momentum constraint satisfied
// by chosen Aij (for now sourced by Bowen York data),
// lapse = 1 shift = 0, phi is the scalar field and is used to
// calculate the rhs, Pi = dphidt = 0.
int poissonSolve(const Vector<DisjointBoxLayout> &a_grids,
                 const PoissonParameters &a_params, Vector<Vector<Vector<Vector<double>>>> rnd_phases)
{
    // the params reader
    ParmParse pp;

    // create the necessary hierarchy of data components
    int nlevels = a_params.numLevels;
    // the user set initial conditions - currently including psi, phi, A_ij
    Vector<LevelData<FArrayBox> *> multigrid_vars(nlevels, NULL);
    // the correction to the conformal factor - what the solver solves for
    Vector<LevelData<FArrayBox> *> dW(nlevels, NULL);
    // the solver vars - coefficients and source
    Vector<LevelData<FArrayBox>*> rhs(nlevels, NULL);
    // the coeff for the I term
    Vector<RefCountedPtr<LevelData<FArrayBox>>> aCoef(nlevels);
    // the coeff for the Laplacian
    Vector<RefCountedPtr<LevelData<FArrayBox>>> bCoef(nlevels);

    // Grid params
    Vector<ProblemDomain> vectDomains(nlevels); // the domains
    Vector<RealVect> vectDx(nlevels); // the grid spacings on each level

    // Set temp vars at coarsest level values
    RealVect dxLev = RealVect::Unit;
    dxLev *= a_params.coarsestDx;
    ProblemDomain domLev(a_params.coarsestDomain);
    IntVect ghosts = 3 * IntVect::Unit;

    // Declare variables here, with num comps, and ghosts for all
    // sources NB - we want output data to have 3 ghost cells to match GRChombo,
    // although not currently needed for 2nd order stencils used here
    for (int ilev = 0; ilev < nlevels; ilev++)
    {
        multigrid_vars[ilev] =
            new LevelData<FArrayBox>(a_grids[ilev], NUM_MULTIGRID_VARS, ghosts);
            
        dW[ilev] = 
            new LevelData<FArrayBox>(a_grids[ilev], 3, ghosts);
            
        rhs[ilev] = 
            new LevelData<FArrayBox>(a_grids[ilev], 3, IntVect::Zero);
        
        aCoef[ilev] = RefCountedPtr<LevelData<FArrayBox>>(
            new LevelData<FArrayBox>(a_grids[ilev], 1, IntVect::Zero));
        bCoef[ilev] = RefCountedPtr<LevelData<FArrayBox>>(
            new LevelData<FArrayBox>(a_grids[ilev], 1, IntVect::Zero));
        vectDomains[ilev] = domLev;
        vectDx[ilev] = dxLev;
        // set initial guess 
        // and values for other multigrid sources - phi and Aij
        set_initial_conditions(*multigrid_vars[ilev], *dW[ilev], 
                                   vectDx[ilev],a_params, rnd_phases);
        // prepare temp dx, domain vars for next level
        dxLev /= a_params.refRatio[ilev];
        domLev.refine(a_params.refRatio[ilev]);
    }
    

    // set up linear operator
    int lBase = 0;
    MultilevelLinearOp<FArrayBox> mlOp;
    BiCGStabSolver<Vector<LevelData<FArrayBox> *>>
        solver; // define solver object

    // default or read in solver params
    int numMGIter = 1;
    pp.query("numMGIterations", numMGIter);
    mlOp.m_num_mg_iterations = numMGIter;

    int numMGSmooth = 4;
    pp.query("numMGsmooth", numMGSmooth);
    mlOp.m_num_mg_smooth = numMGSmooth;

    int preCondSolverDepth = -1;
    pp.query("preCondSolverDepth", preCondSolverDepth);
    mlOp.m_preCondSolverDepth = preCondSolverDepth;

    Real tolerance = 1.0e-7;
    pp.query("tolerance", tolerance);

    int max_iter = 10;
    pp.query("max_iterations", max_iter);

    int max_NL_iter = 4;
    pp.query("max_NL_iterations", max_NL_iter);

    // Iterate linearised Poisson eqn for NL solution
    Real dW_norm = 0.0;
    for (int NL_iter = 0; NL_iter < max_NL_iter; NL_iter++)
    {

        pout() << "Main Loop Iteration " << (NL_iter + 1) << " out of "
               << max_NL_iter << endl;

        //Calculate K
        for (int ilev = 0; ilev < nlevels; ilev++){
        
                    // Fill ghosts
            if (ilev > 0)
            {
                QuadCFInterp quadCFI(a_grids[ilev], &a_grids[ilev - 1],
                                     vectDx[ilev][0], a_params.refRatio[ilev],
                                     NUM_MULTIGRID_VARS, vectDomains[ilev]);
                quadCFI.coarseFineInterp(*multigrid_vars[ilev], *multigrid_vars[ilev - 1]);
            }

            // For intralevel ghosts
            Copier exchange_copier;
            exchange_copier.exchangeDefine(a_grids[ilev], ghosts);
            
            set_K(*multigrid_vars[ilev], *rhs[ilev], a_params,
                       vectDx[ilev], exchange_copier);
        }
        
        pout() << "K determined" << endl;

        // Calculate values for coefficients here - see SetLevelData.cpp
        // for details
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
        
            // Fill ghosts
            if (ilev > 0)
            {
                QuadCFInterp quadCFI(a_grids[ilev], &a_grids[ilev - 1],
                                     vectDx[ilev][0], a_params.refRatio[ilev],
                                     NUM_MULTIGRID_VARS, vectDomains[ilev]);
                quadCFI.coarseFineInterp(*multigrid_vars[ilev], *multigrid_vars[ilev - 1]);
            }

            // For intralevel ghosts
            Copier exchange_copier;
            exchange_copier.exchangeDefine(a_grids[ilev], ghosts);
            
            set_a_coef(*aCoef[ilev], a_params, vectDx[ilev]);
            set_b_coef(*bCoef[ilev], a_params, vectDx[ilev]);
            set_rhs(*rhs[ilev], *multigrid_vars[ilev], vectDx[ilev], a_params, exchange_copier);
        }
        
        pout() << "Coef determined" << endl;

        // set up solver factory
        RefCountedPtr<AMRLevelOpFactory<LevelData<FArrayBox>>> opFactory =
            RefCountedPtr<AMRLevelOpFactory<LevelData<FArrayBox>>>(
                defineOperatorFactory(a_grids, vectDomains, aCoef, bCoef,
                                      a_params));

        // define the multi level operator
        mlOp.define(a_grids, a_params.refRatio, vectDomains, vectDx, opFactory,
                    lBase);

        // set the more solver params
        bool homogeneousBC = true;
        solver.define(&mlOp, homogeneousBC);
        solver.m_verbosity = a_params.verbosity;
        solver.m_normType = 0;
        solver.m_eps = tolerance;
        solver.m_imax = max_iter;


        pout() << "Data output" << endl;

        // Engage!
        for(int i=0; i<3 ; i++)
        {   
            //First copy the components into different levels
            Vector<LevelData<FArrayBox> *> tempRhs(nlevels ,NULL);
            Vector<LevelData<FArrayBox> *> tempdW(nlevels ,NULL);
            for (int level = 0; level < nlevels; level++)
            {
                tempRhs[level] = new LevelData<FArrayBox>(a_grids[level], 1, IntVect::Zero);
                tempdW[level] = new LevelData<FArrayBox>(a_grids[level], 1, ghosts);
                rhs[level]->copyTo(Interval(i,i), *tempRhs[level], Interval(0,0));
                dW[level]->copyTo(Interval(i,i), *tempdW[level], Interval(0,0));
            }
            //Solve
            solver.solve(tempdW, tempRhs);
            
            //Copy back result
            for (int level = 0; level < nlevels; level++)
            {
                tempRhs[level]->copyTo(Interval(0,0), *rhs[level], Interval(i,i));
                tempdW[level]->copyTo(Interval(0,0), *dW[level], Interval(i,i));
            }
            //Clean Temp Levels
            for (int level = 0; level < nlevels; level++)
            {
                if (tempRhs[level] != NULL)
                {
                    delete tempRhs[level];
                    tempRhs[level] = NULL;
                }
                if (tempdW[level] != NULL)
                {
                    delete tempdW[level];
                    tempdW[level] = NULL;
                }
            }
        }
        
                // output the data before the solver acts to check starting conditions
        output_solver_data(dW, rhs, multigrid_vars, a_grids, a_params,
                           NL_iter);
                           
        
        pout() << "Solved Poisson eq" << endl;
        
        
        RealVect dW_sum;
                    for(int i=0; i<=2; i++){
                dW_sum[i] = computeMin(dW, a_params.refRatio, 
                                Interval(i,i))-1.0E-10;
            }
            

        // Add the solution to the linearised eqn to the previous iteration
        // ie W_i -> W_i + dW_i
        // need to fill interlevel and intralevel ghosts first in dpsi
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            
            //norm_dW(*dW[ilev], dW_sum);

            // For interlevel ghosts
            if (ilev > 0)
            {
                QuadCFInterp quadCFI(a_grids[ilev], &a_grids[ilev - 1],
                                     vectDx[ilev][0], a_params.refRatio[ilev],
                                     3, vectDomains[ilev]);
                quadCFI.coarseFineInterp(*dW[ilev], *dW[ilev - 1]);
            }

            // For intralevel ghosts - this is done in set_update_W_i
            // but need the exchange copier object to do this
            Copier exchange_copier;
            exchange_copier.exchangeDefine(a_grids[ilev], ghosts);
            

            // now the update
            set_update_W_i(*multigrid_vars[ilev], *dW[ilev], 
                                exchange_copier);
            for(int i=0; i<=2; i++){
                dW_sum[i] = computeMin(multigrid_vars, a_params.refRatio, 
                                Interval(2+i,2+i))-1.0E-15;
            }
            
            //norm_W(*multigrid_vars[ilev], dW_sum);
        }
        
        pout() << "W_i updated" << endl;
        
        
        
        //Update A_ij
        for (int ilev=0; ilev< nlevels; ilev++)
        {
                    // For interlevel ghosts
            if (ilev > 0)
            {
                QuadCFInterp quadCFI(a_grids[ilev], &a_grids[ilev - 1],
                                     vectDx[ilev][0], a_params.refRatio[ilev],
                                     3, vectDomains[ilev]);
                quadCFI.coarseFineInterp(*dW[ilev], *dW[ilev - 1]);
            }

            // For intralevel ghosts - this is done in set_update_W_i
            // but need the exchange copier object to do this
            Copier exchange_copier;
            exchange_copier.exchangeDefine(a_grids[ilev], ghosts);
            set_A_ij(*multigrid_vars[ilev], *rhs[ilev], vectDx[ilev], a_params, exchange_copier);
        }
        
        pout() << "A_ij updated" << endl; 

        // check if converged or diverging and if so exit NL iteration for loop
        dW_norm = computeNorm(dW, a_params.refRatio, a_params.coarsestDx,
                                Interval(0, 0));
        pout() << "The norm of dW_i after step " << NL_iter + 1 << " is "
               << dW_norm << endl;
        
        /*if (dW_norm < tolerance || dW_norm > 1e5)
        {
            break;
        }*/

    } // end NL iteration loop

    //pout() << "The norm of dW_i at the final step was " << dW_norm << endl;

    // Mayday if result not converged at all - using a fairly generous threshold
    // for this as usually non convergence means everything goes nuts
    /*if (dW_norm > 1e-1)
    {
        MayDay::Error(
            "NL iterations did not converge - may need a better initial guess");
    }*/
    
    //Final Ham and Mom Constraint
    Vector<LevelData<FArrayBox>*> constraints (nlevels, NULL);
    for (int ilev = 0; ilev < nlevels; ilev++)
    {
        constraints[ilev] =
            new LevelData<FArrayBox>(a_grids[ilev], 4, IntVect::Zero);
        get_Const(*multigrid_vars[ilev], *constraints[ilev], a_params, vectDx[ilev]);
    } 
    
    Real hamConst = computeNorm(constraints, a_params.refRatio, a_params.coarsestDx,
                                    Interval(3, 3));
    Real momConst = computeNorm(constraints, a_params.refRatio, a_params.coarsestDx,
                                    Interval(0, 2));
    pout() << "FINAL HAMILTONIAN CONSTRAINT: " << hamConst << endl;
    pout() << "FINAL MOM CONSTRAINT: " << momConst << endl;
    
    // now output final data in a form which can be read as a checkpoint file
    // for the GRChombo AMR time dependent runs
    output_final_data(multigrid_vars, a_grids, vectDx, vectDomains, a_params);

    // clean up data
    for (int level = 0; level < multigrid_vars.size(); level++)
    {
        if (multigrid_vars[level] != NULL)
        {
            delete multigrid_vars[level];
            multigrid_vars[level] = NULL;
        }
        if (rhs[level] != NULL)
        {
            delete rhs[level];
            rhs[level] = NULL;
        }
        if (dW[level] != NULL)
        {
            delete dW[level];
            dW[level] = NULL;
        }
    }
    int exitStatus = solver.m_exitStatus;
    // note that for AMRMultiGrid, success = 1.
    exitStatus -= 1;
    return exitStatus;
}

// Main function - keep this simple with just setup and read params
int main(int argc, char *argv[])
{
            
        
    int status = 0;
    int p, my_rank;
#ifdef CH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    // Scoping trick
    {
        if (argc < 2)
        {
            cerr << " usage " << argv[0] << " <input_file_name> " << endl;
            exit(0);
        }

        char *inFile = argv[1];
        ParmParse pp(argc - 2, argv + 2, NULL, inFile);

        PoissonParameters params;
        Vector<DisjointBoxLayout> grids;

        // read params from file
        getPoissonParameters(params);
        
        
        //get random phases for phi
        
        Vector<Vector<Vector<Vector<double>>>> rnd_phases(params.nCells[0], Vector<Vector<Vector<double>>>(params.nCells[1], Vector<Vector<double>>(params.nCells[2], Vector<double>(2, 0.0))));
        if(my_rank == 0){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double m2 = 3*pow(10, -4)*pow(params.phi_0,2);
        for(int i=-params.nCells[0]/2; i < params.nCells[0]/2;i++){
            for(int j=-params.nCells[0]/2; j < params.nCells[0]/2;j++){
                for(int k=-params.nCells[0]/2; k < params.nCells[0]/2; k++){
                    double aux = dis(gen);
                    rnd_phases[i+params.nCells[0]/2][j+params.nCells[0]/2][k+params.nCells[0]/2][1]=aux*2*M_PI;
                    
                    double waven = 2*M_PI*sqrt(i*i+j*j+k*k)/params.domainLength[0];
                    double sigma = pow(params.nCells[0],3)*pow(params.domainLength[0], -3/2)/pow((pow(waven,2)+m2),0.5)/sqrt(M_PI)*params.phi_amplitude;
                    //aux = dis(gen);
                    rnd_phases[i+params.nCells[0]/2][j+params.nCells[0]/2][k+params.nCells[0]/2][0] = sigma;//*sqrt(-2*log(aux));
                    if(i==j && j==k && k == 0){
                       rnd_phases[i+params.nCells[0]/2][j+params.nCells[0]/2][k+params.nCells[0]/2][0] = 0;
                    }
                }
            }
        }
        }
        pout() << "Random phases" << endl;
        /*for(int i = 0; i < params.nCells[]; i++){
        for(int j = 0; j < params.nCells[]; j++){
        for(int k = 0; k < params.nCells[]; k++){
            double k = 2*M_PI*sqrt((i+1)*(i+1)+(j+1)*(j+1)+(k+1)*(k+1))
            double sigma = sqrt(pow(1/params.L[0], 3)*params.phi_amplitude/(M_PI*k));
            std::rayleigh_distribution<double> rayleigh(sigma);
            rnd_amplitudes[i][j][k][0] = rayleigh(gen);
        }
        }
        }*/

        // set up the grids, using the rhs for tagging to decide
        // where needs additional levels
        
        set_grids(grids, params);

        // Solve the equations!
        status = poissonSolve(grids, params, rnd_phases);

    } // End scoping trick

#ifdef CH_MPI
    MPI_Finalize();
#endif
    return status;
}

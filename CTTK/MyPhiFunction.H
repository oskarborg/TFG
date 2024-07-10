/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MYPHIFUNCTION_HPP
#define MYPHIFUNCTION_HPP

#define Nmodes 8

Real my_phi_function(RealVect loc, Real amplitude, IntVect N, RealVect L, Real phi_0, Vector<Vector<Vector<Vector<double>>>> rnd_phases)
{

    Real bkg_phi = phi_0;
    for(int i=-Nmodes/2; i < Nmodes/2;i++){
            for(int j=-Nmodes/2; j < Nmodes/2;j++){
                for(int k=-Nmodes/2; k< Nmodes/2; k++){
                    bkg_phi += rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][0]*cos(2*M_PI * ( loc[0] * i + loc[1]*j + loc[2]* k)/ L[0]+rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][1])*pow(Nmodes, -3)/2;
                    bkg_phi += rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][0]*cos(-2*M_PI * ( loc[0] * i + loc[1]*j + loc[2]* k)/ L[0]+rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][1])*pow(Nmodes, -3)/2;
                }
            }
    }
    return bkg_phi;
}

Real my_Pi_function(RealVect loc, Real amplitude, IntVect N, RealVect L, Real phi_0, Real Pi_0, Real phi_here, Vector<Vector<Vector<Vector<double>>>> rnd_phases)
{

    // sine waves in 3 directions - use with periodic BCs
    Real bkg_Pi = Pi_0;
    Real H=sqrt((0.5*pow(Pi_0,2)+pow(10,-4)*pow(phi_0, 4)/4)/3);
    Real m2 = 3*pow(10, -4)*pow(phi_0,2);
    for(int i=-Nmodes/2; i < Nmodes/2;i++){
            for(int j=-Nmodes/2; j < Nmodes/2;j++){
                for(int k=-Nmodes/2; k< Nmodes/2; k++){
                
                    double waven = 2*M_PI*sqrt(i*i+j*j+k*k)/L[0];
                    double amplitude = rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][0]*pow((pow(waven,2)+m2),0.5);
                    //bkg_Pi -= amplitude*sin(2*M_PI * ( loc[0] * i + loc[1]*j + loc[2]* k)/ L[0]+rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][1])*pow(Nmodes, -3)/2;
                    //bkg_Pi += amplitude*sin(-2*M_PI * ( loc[0] * i + loc[1]*j + loc[2]* k)/ L[0]+rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][1])*pow(Nmodes, -3)/2;
                    bkg_Pi += H * rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][0]*cos(2*M_PI * ( loc[0] * i + loc[1]*j + loc[2]* k)/ L[0]+rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][1])*pow(Nmodes, -3)/2;
                    bkg_Pi += H * rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][0]*cos(-2*M_PI * ( loc[0] * i + loc[1]*j + loc[2]* k)/ L[0]+rnd_phases[i+N[0]/2][j+N[0]/2][k+N[0]/2][1])*pow(Nmodes, -3)/2;
                }
            }	
    }
    return bkg_Pi;
}

#endif /* MYPHIFUNCTION_HPP */

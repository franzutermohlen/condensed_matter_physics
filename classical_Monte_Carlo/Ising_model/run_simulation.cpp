// This code performs a classical Monte Carlo simulation the Ising model on a square lattice.
//
// Copyright © 2021 Franz Utermohlen. All rights reserved.

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include </usr/local/include/eigen3/Eigen/Eigen> // My Eigen directory

// using namespace std;
using namespace Eigen;

#define PI 3.14159265

// ===================Global constants===================

// **********INPUTS**********
// Number of sites along one of the orthogonal directions
const int L = 3;

// Exchange coupling J
const double J = -3.48;

// External magnetic field h (only works when using single-site updates)
double h = 0;

// Number of temperature vals to probe
const int numTvals = 50;
// **************************

// Number of unit cells in the lattice along the two orthogonal directions
const int Nx = L;
const int Ny = L;

// Declare the total number of sites N and the temperature T
int N;
double T;

// ======================================================

void find_neighbors_square_lattice(int i, int j, ArrayXXi &neighborList) {

  // Note: neighborList(k,0) contains the i index of the kth neighbor, and neighborList(k,1) contains the j index of the kth neighbor

  // 1st neighbor
  neighborList(0,0) = (i - 1 + Nx) % Nx;
  neighborList(0,1) = j;
  // 2nd neighbor
  neighborList(1,0) = (i + 1) % Nx;
  neighborList(1,1) = j;
  // 3rd neighbor
  neighborList(2,0) = i;
  neighborList(2,1) = (j - 1 + Ny) % Ny;
  // 4th neighbor
  neighborList(3,0) = i;
  neighborList(3,1) = (j + 1) % Ny;

}

void Metropolis_sweep_Ising(Array<int,Nx,Ny> &configIsing, double &MCurrent) {

  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {

      // Original spin value at site (i,j)
      int spinOrig = configIsing(i,j);

      // Calculate the original bond energies corresponding to site (i,j)
      double neighboringBondEnergiesOrig = 0;

      // Declare neighborList
      ArrayXXi neighborList(4,2);

      // Find the 4 nearest neighbors
      find_neighbors_square_lattice(i, j, neighborList);

      for (int neighbor = 0; neighbor < 4; neighbor++) {

        int ineighbor = neighborList(neighbor,0);
        int jneighbor = neighborList(neighbor,1);

        // Add the energy from the current bond
        neighboringBondEnergiesOrig += J * spinOrig * configIsing(ineighbor,jneighbor);

      }

      // Trial (after the spin flip) bond energies and Zeeman energy corresponding to site (i,j)
      double neighboringBondEnergiesTrial = -neighboringBondEnergiesOrig;
      // Zeeman energy
      double ZeemanEnergyOrig = h * spinOrig;
      double ZeemanEnergyTrial = -ZeemanEnergyOrig;

      // Calculate the energy cost deltaE = E_trial - E_old for making this spin flip
      double deltaE = (neighboringBondEnergiesTrial + ZeemanEnergyTrial) - (neighboringBondEnergiesOrig + ZeemanEnergyOrig);

      // Initialize flip; if flip = true after the following if statement, we will perform the flip
      bool flip = false;

      // If deltaE <= 0, make the flip. Otherwise, if deltaE > 0, flip the spin with a probability e^(-deltaE/(k_B T))
      if (deltaE <= 0) {
        flip = true;
      }
      else if ((double)rand()/RAND_MAX <= exp(-deltaE / T)) {
        flip = true;
      }

      if (flip == true) {
        // Flip the spin at (i,j)
        configIsing(i,j) *= -1;
        // Update the system's magnetization
        MCurrent += 2 * configIsing(i,j);
      }

    }
  }

}

// void grow_cluster_Ising(int i, int j, int spinOrig, Array<int,Nx,Ny> &configIsing, double &MCurrent) {
//
//   // Flip the spin at (i,j)
//   configIsing(i,j) *= -1;
//
//   // Update the system's magnetization
//   // MCurrent += 2 * configIsing(i,j);
//   MCurrent -= 2 * spinOrig;
//
//   // Declare neighborList
//   ArrayXXi neighborList(4,2);
//
//   // Find the 4 nearest neighbors
//   find_neighbors_square_lattice(i, j, neighborList);
//
//   // Loop over neighbors
//   for (int neighbor = 0; neighbor < 4; neighbor++) {
//
//     int ineighbor = neighborList(neighbor,0);
//     int jneighbor = neighborList(neighbor,1);
//
//     // Test neighbor to see if it joins the cluster
//     if ( configIsing(ineighbor,jneighbor) == spinOrig && (double)rand()/RAND_MAX <= (1. - exp(2.0 * J / T)) ) {
//       grow_cluster_Ising(ineighbor, jneighbor, spinOrig, configIsing, MCurrent);
//     }
//
//   }
//
// }

int main() {

  // Total number of sites
  N = Nx * Ny;

  // External magnetic field h (only works when using the Metropolis algorithm)
  // double h = 0;

  // ==================================

  // Starting time (in order to print the runtime at the end)
  clock_t t1,t2;
  t1 = clock();

  // Set up the output files
  const int numOutputFiles = 3; // Number of output files
  std::string filename[numOutputFiles];
  filename[0] = "data_m_";
  filename[1] = "data_susc_";
  filename[2] = "data_Binder_";
  std::string Nxstr, Nystr, fileformat, modelname;
  modelname = "Ising";
  Nxstr = std::to_string(Nx);
  Nystr = std::to_string(Ny);
  fileformat = ".txt";

  std::ofstream outputFile[numOutputFiles];
  for (int i = 0; i < numOutputFiles; i++) {
    filename[i] += modelname + '_' + Nxstr + 'x' + Nystr + fileformat;
    outputFile[i].open(filename[i].c_str());
    outputFile[i] << std::fixed; // Don't use scientific notation in the output file
  }
  // outputFile[0] << "datam" + Nxstr + 'x' + Nystr + " = {";
  // outputFile[1] << "datasusc" + Nxstr + 'x' + Nystr + " = {";
  // outputFile[2] << "dataBinder" + Nxstr + 'x' + Nystr + " = {";
  outputFile[0] << '{';
  outputFile[1] << '{';
  outputFile[2] << '{';

  // Seed the random number sequence
  srand(time(NULL));

  // Number of time steps per data point at a given temperature value
  unsigned long numSteps = ceil( pow(N,2.5) );
  // unsigned long numSteps = pow(10,5);

  // Dynamical critical exponent for this algorithm (which is approx. 0.45 for cluster updates and 2 for Metropolis sweeps)
  // double z = 2;
  double z = 1;
  // double z = 0.5;

  // Estimate the maximum autocorrelation time
  int tauMax = pow(L,z);
  int tau = ceil(tauMax);

  // Number of measurements to discard from the beginning of each calculation at a new temperature (usually somewhere from 5 to 10)
  const int numMeasurementsToDiscard = 0;//5;

  // Number of measurements to average when calculating thermal expectation values
  const unsigned long numMeasurementsToAvg = (numSteps/tau) - numMeasurementsToDiscard;

  // Min and max temperature vals to probe
  double Tmin = 0.01;
  double Tmax;
  Tmax = 4.5;

  // Separation between temperature vals
  double deltaT = 0;
  if (numTvals > 1) {
    deltaT = (Tmax - Tmin)/(numTvals - 1);
  }

  // Optional: Declare arrays that will save the information about T, m, susc, and Binder cumulant
  // std::array<double,numTvals> Tlist;
  // std::array<double,numTvals> mlist;
  // std::array<double,numTvals> susclist;
  // std::array<double,numTvals> Binderlist;

  // Configuration array
  Array<int,Nx,Ny> configIsing;

  // The structure of this array is (i,j), where i and j are coordinates in the square lattice, and its dimensions are (Nx,Ny). Each entry is either +1 or -1 (spin up or spin down).
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      configIsing(i,j) = 2 * (rand() % 2) - 1;
    }
  }

  for (int Tindex = 0; Tindex < numTvals; Tindex++) {

    // Temperature val for this data point
    // Going from low T to high T
    // T = Tmin + Tindex * deltaT;
    // Going from high T to low T
    T = Tmax - Tindex * deltaT;

    // // Configuration array
    // Array<int,Nx,Ny> configIsing;
    //
    // // The structure of this array is (i,j), where i and j are coordinates in the square lattice, and its dimensions are (Nx,Ny). Each entry is either +1 or -1 (spin up or spin down).
    // for (int i = 0; i < Nx; i++) {
    //   for (int j = 0; j < Ny; j++) {
    //     configIsing(i,j) = 2 * (rand() % 2) - 1;
    //   }
    // }

    // Declare MCurrent
    double MCurrent;

    // Find the system's initial value of M
    MCurrent = configIsing.sum();

    // Declare M, M2, M4, which are the expectation values <|M|>, <M^2>, and <M^4>, respectively, where M = sum_i s_i
    double M = 0;
    double M2 = 0;
    double M4 = 0;

    // Iterate algorithm numSteps times
    for (unsigned long t = 0; t < numSteps; t++) {

      // Pick a random site (iR,jR)
      int iR = rand() % Nx;
      int jR = rand() % Ny;

      // Sweep through the entire lattice performing single-site updates
      Metropolis_sweep_Ising(configIsing, MCurrent);

      // Uncomment the following 2 lines to perform a cluster update about site (iR,jR)
      // int spinOrig = configIsing(iR,jR);
      // grow_cluster_Ising(iR, jR, spinOrig, configIsing, MCurrent);

      // Find the value of M^2 for the current time step
      if (t >= numMeasurementsToDiscard*tau && (t+1) % tau == 0) {
        double M2Current = pow(MCurrent,2);
        double M4Current = pow(MCurrent,4);

        M += std::abs(MCurrent);
        M2 += M2Current;
        M4 += M4Current;
      }
    }

    // Take the thermal average of M, M^2, and M^4
    M /= numMeasurementsToAvg;
    M2 /= numMeasurementsToAvg;
    M4 /= numMeasurementsToAvg;

    // Get m
    double m = M / N;

    // Optional: Get m^2
    // double m2 = M2 / (N * N);

    // Determine the susceptibility
    double susc = (1/T) * (M2 - M * M);

    // Determine the fourth-order Binder cumulant (or Binder ratio), which is 1/2[3 - <M^4>/(<M^2>^2)]
    double Binder = 1./2. * ( 3. - M4/(M2 * M2) );

    // Optional: Save these values of T, m, and susc (if we made these arrays)
    // Tlist[Tindex] = T;
    // mlist[Tindex] = m;
    // susclist[Tindex] = susc;
    // Binderlist[Tindex] = Binder;

    // Save these values to their respective data files
    if (Tindex < numTvals - 1) {
      outputFile[0] << "{" << T << ", " << m << "},\n";
      outputFile[1] << "{" << T << ", " << susc << "},\n";
      outputFile[2] << "{" << T << ", " << Binder << "},\n";
    }
    else {
      outputFile[0] << "{" << T << ", " << m << "}}";
      outputFile[1] << "{" << T << ", " << susc << "}}";
      outputFile[2] << "{" << T << ", " << Binder << "}}";
    }
  }

  // Close output streams
  for (int i = 0; i < numOutputFiles; i++) {
    outputFile[i].close();
  }

  // Print the runtime
  t2 = clock();
  double diff = t2 - t1;
  double seconds = diff / CLOCKS_PER_SEC;
  std::cout << "runtime = " << seconds << " s" << std::endl;

  return 0;
}

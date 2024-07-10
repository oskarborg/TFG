//Calcular epsilon y N en cada punto del espacio

#ifndef COMPUTEEPS_HPP_
#define COMPUTEEPS_HPP_

#include "Cell.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include "CCZ4Geometry.hpp"
#include "simd.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "Potential.hpp"


class ComputeEps
{
  public:
  
   ComputeEps(Potential::params_t a_params) : params(a_params){}
   
template <class data_t> struct Vars
    {
        data_t phi;
        data_t Pi;
        data_t epsilon;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function);
    };

		
   template <class data_t> void compute(Cell<data_t> current_cell) const
    {
      auto vars = current_cell.template load_vars<Vars>();
      
      Potential potential(params);

      data_t V, dV;
      potential.compute_potential(V, dV, vars);
      vars.epsilon=3*pow(vars.Pi,2)/(pow(vars.Pi,2)+2*V);

      current_cell.store_vars(vars);
    }
  protected:
    Potential::params_t params;
};

template <class data_t>
template <typename mapping_function_t>
void ComputeEps::Vars<data_t>::enum_mapping(
    mapping_function_t mapping_function)
{
    VarsTools::define_enum_mapping(mapping_function, c_phi, phi);
    VarsTools::define_enum_mapping(mapping_function, c_Pi, Pi);
    VarsTools::define_enum_mapping(mapping_function, c_epsilon, epsilon);
}
#endif

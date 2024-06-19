//Calcular epsilon y N en cada punto del espacio

#ifndef COMPUTEVARIANCE_HPP_
#define COMPUTEVARIANCE_HPP_

#include "Cell.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include "CCZ4Geometry.hpp"
#include "simd.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"


class ComputeVariance
{
  public:
  
   ComputeVariance(double phi) : avg_phi(phi){}
   
template <class data_t> struct Vars
    {
        data_t phi;
        data_t variance;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function);
    };

		
   template <class data_t> void compute(Cell<data_t> current_cell) const
    {
      auto vars = current_cell.template load_vars<Vars>();
      
      vars.variance=pow((vars.phi-avg_phi),2);

      current_cell.store_vars(vars);
    }
  protected:
    double avg_phi;
};

template <class data_t>
template <typename mapping_function_t>
void ComputeVariance::Vars<data_t>::enum_mapping(
    mapping_function_t mapping_function)
{
    VarsTools::define_enum_mapping(mapping_function, c_phi, phi);
    VarsTools::define_enum_mapping(mapping_function, c_variance, variance);
}
#endif

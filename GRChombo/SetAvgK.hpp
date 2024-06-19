//Calcular epsilon y N en cada punto del espacio

#ifndef SETAVGK_HPP_
#define SETAVGK_HPP_

#include "Cell.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include "CCZ4Geometry.hpp"
#include "simd.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "Potential.hpp"


class SetAvgK
{
  public:
  
   SetAvgK(double a_davgK) : davgK(a_davgK){}
   
template <class data_t> struct Vars
    {
        data_t avgK;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function);
    };

		
   template <class data_t> void compute(Cell<data_t> current_cell) const
    {
      auto vars = current_cell.template load_vars<Vars>();
      
      vars.avgK = davgK;

      current_cell.store_vars(vars);
    }
  protected:
    double davgK;
};

template <class data_t>
template <typename mapping_function_t>
void SetAvgK::Vars<data_t>::enum_mapping(
    mapping_function_t mapping_function)
{
    VarsTools::define_enum_mapping(mapping_function, c_avgK, avgK);
}
#endif

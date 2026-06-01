#ifndef MADNESS_APPS_MOLDFT_FCWF_H_INCLUDED
#define MADNESS_APPS_MOLDFT_FCWF_H_INCLUDED

#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <vector>
#include <math.h>
#include <complex>

using namespace madness;

class Fcwf{
     std::vector<complex_function_3d> m_psi;
     double speed_of_light;
     double s_fac;
     bool m_initialized;

public:

     Fcwf();

     Fcwf(const complex_function_3d& wf1,
          const complex_function_3d& wf2,
          const complex_function_3d& wf3,
          const complex_function_3d& wf4,
          const double myc,
          const double my_s_fac);

     Fcwf(World& world, const double myc, const double my_s_fac);

     complex_function_3d& operator[](const int i);

     const complex_function_3d& operator[](const int i) const ;

     explicit Fcwf(std::vector<complex_function_3d>& phi, const double myc, const double my_s_fac);

     bool getinitialize();

     bool getinitialize() const ;

     double get_myc();

     double get_myc() const ;

     double get_small_comp_factor();

     double get_small_comp_factor() const ;

     unsigned int size();

     unsigned int size() const ;

     Fcwf(const Fcwf& phi, const double myc, const double my_s_fac);

     Fcwf operator=(const Fcwf& phi);

     Fcwf operator-(const Fcwf& phi) const ;

     Fcwf operator+(const Fcwf& phi);

     Fcwf operator*(std::complex<double> a) const ;
     
     void scale(std::complex<double> a);
     
     Fcwf operator+=(const Fcwf& phi);

     Fcwf operator-=(const Fcwf& phi);

     double norm2();

     void normalize();

     Fcwf operator*(madness::complex_function_3d& phi);

     Fcwf operator*(madness::real_function_3d& phi);

     void truncate();

     std::complex<double>  inner(World& world, const Fcwf& phi) const;

     void apply(World& world, real_convolution_3d& op);

     void apply(World& world, complex_derivative_3d& D);
     
     void reconstruct();

     void compress();

     Fcwf KramersPair();

};

std::complex<double> inner(const Fcwf& psi, const Fcwf& phi);

Fcwf apply(World& world, real_convolution_3d& op, const Fcwf& psi);

Fcwf apply(World& world, complex_derivative_3d& op, const Fcwf& psi);

real_function_3d squaremod(Fcwf& psi);

real_function_3d squaremod_small(Fcwf& psi);

real_function_3d squaremod_large(Fcwf& psi);

complex_function_3d inner_func(World& world, Fcwf& psi, Fcwf& phi);

Fcwf copy(Fcwf psi);

std::complex<double> inner(std::vector<Fcwf>& a, std::vector<Fcwf>& b);

std::vector<Fcwf> operator*(const std::vector<Fcwf>& psis, std::complex<double> a);

std::vector<Fcwf> operator*(std::complex<double> a, const std::vector<Fcwf>& psis);

void operator+=(std::vector<Fcwf>& phi, const std::vector<Fcwf>& psi);

std::vector<Fcwf> operator-(const std::vector<Fcwf>& phi, const std::vector<Fcwf>& psi);

Tensor<std::complex<double>> matrix_inner(World& world, std::vector<Fcwf>& a, std::vector<Fcwf>& b);

std::vector<Fcwf> transform(World& world, std::vector<Fcwf>& a, Tensor<std::complex<double>> U);

//allocator class needed for KAIN
class Fcwf_vector_allocator {
     World& world;
     unsigned int m_size;
     double speed_of_light;
     double s_fac;
     public:
          //Constructor
          Fcwf_vector_allocator(World& world, unsigned int m_size, const double& myc, const double& my_s_fac);

          //Overloading () operator
          std::vector<Fcwf> operator()();

          //Copy Constructor
          Fcwf_vector_allocator operator=(const Fcwf_vector_allocator& other);

          void set_size(int size);
};

#endif

//kthxbye

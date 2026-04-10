#ifndef minus_hxx_
#define minus_hxx_
// 
// \brief MInimal problem NUmerical continuation package
// \author Ricardo Fabbri based on original code by Anton Leykin 
// \date Created: Fri Feb  8 17:42:49 EST 2019
// 
#include <cstdio>
#include <iomanip>
#include <cstring>
#include "minus.h"
#include "internal-util.hxx"

//#define M_VERBOSE  // TODO: cmake option like INTERNAL TELEMETRY ON
#include "debug-util.h"

// Gabriel (12-06-2025): Add stuff for test
#include <fstream>
#include "Eigen/Dense"

#define unlikely(expr) __builtin_expect(!!(expr),0)
#define likely(expr)   __builtin_expect(!!(expr),1)
// TODO: perhaps move inside problem.hxx
#include "chicago14a-lsolve.hxx"
// #include "partialpivLU-NxN-lsolve.hxx" // TODO put generic solve here
#include "linecircle2a-lsolve.hxx"

namespace MiNuS {


// THE MEAT //////////////////////////////////////////////////////////////////////
//
// INPUT
//    s: tracker settings
//    s_sols: start sols      
//    params: parameters of the homotopy between start and end system. This is currently the
//            parameters of the start system, followed by that of of target system as
//            specialized homotopy params (P01 in tutorial and SolveChicago)
//    
// OUTPUT
//    raw_solutions: compute solutions sol_min...sol_max-1 within all nsols
// 
template <problem P, typename F> void 
__attribute__((no_sanitize("address")))
minus_core<P, F>::
track(const track_settings &s, 
      const C<F> s_sols_u[f::nve*f::nsols], 
      const C<F> params_u[2*f::nparams], 
      solution raw_solutions_u[f::nsols], 
      unsigned sol_min, unsigned sol_max)
{
  const C<F> *s_sols = reinterpret_cast<C<F> *> (__builtin_assume_aligned(s_sols_u,64));
  const C<F> *params = reinterpret_cast<C<F> *> (__builtin_assume_aligned(params_u,64));

  solution *raw_solutions = reinterpret_cast<solution *> (__builtin_assume_aligned(raw_solutions_u,64));
  assert(sol_min <= sol_max && sol_max <= f::nsols);
  alignas(64) C<F> Hxt[NVEPLUS1 * f::nve]; 
  alignas(64) F x0t0f[f::nve*2+1];
  alignas(64) F xtf[f::nve*2+1];
  alignas(64) F dxdtf[f::nve*2+1];
  alignas(64) C<F> dxi[f::nve];
  C<F> *const x0t0 = (C<F> *) x0t0f;
  C<F> *const xt = (C<F> *) xtf;
  C<F> *const dxdt = (C<F> *) dxdtf;
  C<F> *const x0 = x0t0;
  F    *const t0 = (F *) (x0t0 + f::nve);
  F    *const t  = (F *) (xt + f::nve);
  C<F> *const x1t1 = xt;
  C<F> *const dx = dxdt;
  C<F> *const dx4 = dx;
  F    *const dt = (F *)(dxdt + f::nve);
  C<F> *const HxH=Hxt;
  Map < Matrix<C<F>, f::nve, NVEPLUS1 >, Aligned>
      AA((C<F> *)Hxt, f::nve, NVEPLUS1); // Full Jacobian matrix (also reused for [Hx|H]) in Eigen format
  static constexpr F the_smallest_number = 1e-13; // XXX BENCHMARK THIS
  typedef minus_array<f::nve,F> v;
  typedef numeric_subroutines<P,F> numerics;

  //  alignas(64) C<F> ycHxt[16]; 
  //  alignas(64) C<F> ycHxH[13];
  // memoization_init() : 
  //\Gabriel (added on 12-12-2025--> made by Gabriel A.)
  //\brief: Thresholding volume and condition number.
  //\Sirves to indicate if an solution f(x,t) = 0 is "sucked" to a
  //\singularity. Initial experiments started on (12-13-2025) indicated
  //\that volumes (det(Hx|_(x_0,t_0)) < 1e-33 and condition numbers
  //\(sigma_max/sigma_min) > 16 generates ill conditioned solutions at the
  //\iteration, leading to a singularity. Using as a basis, the idea now is
  //\reduce these numbers to investigate how much can we reduce the time and find
  //\the solution in the worst scenario.
  //(12-18-2025)
  //\MiNuS bin Chicacago on worst scenario the volume and (highly controlled
  //\experiment) I was able to obtain results on GT for volume threshold of
  //1e-32 and condition number threshold of 1e^12.
  //XXX TODO: Decrease and see on openMVG. Experiment Use Capitol high dataset
  //and see if I can get same than published on ENMC 2023. current max core temp
  //was to 80ºC from 100ºC
  //\Tests on Ctest (make test) works fine.
  std::complex<double> jv;
  double jacobian_volume; // Det|Hx| at curr step
  double condition_number; // c at curr step
  double volthresh = 1e-5; // det|jac|_(x0,t0+dt) approx 0. 1e-32 works on
                            // highly controlled but not in real environment
  double cond_thresh = 1e2; // sigma_max/sigma_min too high at step. 1e12 works on
                            // highly controlled but not in real environment
  //double volthresh = 1e-32;
  //double cond_thresh = 1e12;
  double epsilon_svd = 1e-6; // sigma_i too low
  // C<F> previous[13];
  const F &t_step = s.init_dt_;  // initial step
  solution *t_s = raw_solutions + sol_min;  // current target solution
  const C<F>* __restrict s_s = s_sols + sol_min*f::nve;    // current start solution
  for (unsigned sol_n = sol_min; sol_n < sol_max; ++sol_n) { // solution loop
    LLOG("solution  id " << sol_n << "---------------------------------------------------\n"); 
    t_s->status = PROCESSING;
    bool end_zone = false;
    v::copy(s_s, x0);
    *t0 = 0; *dt = t_step;
    
#ifdef M_VERBOSE
    LLOG("H must evaluate to 0 at start solution\n"); 
    evaluate_HxH(x0t0, params, HxH);
    LLOG(AA << std::endl);
    
#if 0
    C<F> gt[f::nve*f::nsols] = { // linecircle debug
      // solution 1
      {-.8399999999999715, -.3803288051147313}, // x
      {.03000000000008518, -1.140986415344194}  // y
      // we dont include more sols now, just want it to find the one above
      // TODO: make it real for your problem, to be more realistic in profiling
    };
   j 
    LLOG("H|t=1 must evaluate to 0 at GT solution\n"); 
    v::copy(gt, x0);
    *t0 = 1;
    LLOG("x0t0\n"); 
    pprint((F *)x0t0, f::nve*2+1, true);
    LLOG("H\n"); 
    evaluate_HxH(x0t0, params, HxH);
    std::cerr << AA << std::endl;

    *t0 = 0;
    // evaluate_HxH(x0t0, params, HxH);
#endif
#endif
    
    char predictor_successes = 0;
    //std::cout << sol_min << std::endl;
    // XXX print the values of H
    // everywhere

    // track H(x,t) for t in [0,1]
    // TODO: due to precision, it is best from 1 to 0 as Bertini/Wampler suggests
    while (likely(t_s->status == PROCESSING && 1. - *t0 > the_smallest_number)) {
      if (unlikely(t_s->num_steps == s.max_num_steps_)) {
        t_s->status = MAX_NUM_STEPS_FAIL; // failed to reach solution in the available step budget
        break;
      }
      
      if (unlikely(!end_zone && 1. - *t0 <= s.end_zone_factor_ + the_smallest_number))
        end_zone = true; // TODO: see if this path coincides with any other path on entry to the end zone
      if (unlikely(end_zone)) {
          if (unlikely(*dt > 1. - *t0)) *dt = 1 - *t0;
      } else if (unlikely(*dt > 1. - s.end_zone_factor_ - *t0)) *dt = 1. - s.end_zone_factor_ - *t0;
      /// PREDICTOR ------------------------------------------------------------
      //    in: x0t0, dt 
      //    out: dx
      //    
      /*  high-level code for Runge-Kutta-4
          dx1 := solveHxTimesDXequalsminusHt(x0,t0);
          dx2 := solveHxTimesDXequalsminusHt(x0+(1/2)*dx1*dt,t0+(1/2)*dt);
          dx3 := solveHxTimesDXequalsminusHt(x0+(1/2)*dx2*dt,t0+(1/2)*dt);
          dx4 := solveHxTimesDXequalsminusHt(x0+dx3*dt,t0+dt);
          (1/6)*dt*(dx1+2*dx2+2*dx3+dx4) */
      v::fcopy(x0t0, xt);

      // dx1 ----
      // evaluate_Hxt_constants(xt, params, ycHxt);
      memoize_Hxt(Hxt);/*, ycHxt);*/
      evaluate_Hxt(xt, params, Hxt); // Outputs full Jacobian matrix Hxt
      // dx4_eigen = lu.compute(AA).solve(bb);
      // Gabriel: test hardcoded
      // Gabriel (09-12-2025): Output into file to study for now since Eigen is being a pussy!!
      // Gabriel Saving each preditor dx1
#if 0
      if(sol_n == sol_min) {
        MatrixXcd vol = AA; // Copy Hxt at first
        vol = vol.block<14,14>(0,0);
        //for (unsigned i = 0; i < 14; ++i) 
        //  for (unsigned j = 0; j < 14; ++j)
        //    vol(i,j) = AA(i,j);
        std::ofstream outf("Hxt_matrix_output_at_step_0_dx1.txt");
        outf << vol << std::endl;
        outf.close();
        JacobiSVD<MatrixXcd, FullPivHouseholderQRPreconditioner> svd(vol);
        VectorXd singval = svd.singularValues();
        std::cout << "Its singular values are:" << std::endl << singval << std::endl;
        std::cout << std::endl<< "Volume of dx1 at first iteration is:" << std::endl << singval.prod() << std::endl<< std::endl;
        std::cout << std::endl<< "Condition number of dx1 at first iteration is:" << std::endl << singval(0)/singval(13) << std::endl<< std::endl;

      }
#endif
      // Gabriel (12-12-2025): Experimental setup threshold parameters
      //{
      //MatrixXcd vol = AA; // Copy Hxt at first
      //vol = vol.block<14,14>(0,0); // Pick just Hx
      //JacobiSVD<MatrixXcd, FullPivHouseholderQRPreconditioner> svd(vol);
      //VectorXd singval = svd.singularValues(); // perform SVD
      //// Zero sigma if below threshold
      //for (unsigned i = 0; i < 14; ++i)
      //  if (singval(i) < epsilon_svd)
      //    singval(i) = 0;
      //jacobian_volume = singval.prod(); // Det|Hx| at curr step
      //condition_number = singval(0)/singval(13); // c at curr step
      ////if (jacobian_volume < volthresh && condition_number > cond_thresh){ // If blows up,
      ////  std::cout << "ill-conditioned solution\n";                       // goes to the next
      ////  break;
      ////}
      //}
      numerics::lsolve(AA, dx4);
      
      // dx2 ----
      const F one_half_dt = *dt*0.5;
      v::multiply_scalar_to_self(dx4, one_half_dt);
      v::add_to_self(xt, dx4);
      v::multiply_scalar_to_self(dx4, 2.);
      *t += one_half_dt;  // t0+.5dt
      evaluate_Hxt(xt, params, Hxt);
#if 0
      // Gabriel Saving each preditor dx2
      if(sol_n == sol_min) {
        Matrix<std::complex<double>, 14, 14> vol;
        for (unsigned i = 0; i < 14; ++i) 
          for (unsigned j = 0; j < 14; ++j)
            vol(i,j) = AA(i,j);
        std::ofstream outf("Hxt_matrix_output_at_step_0_dx2.txt");
        outf << vol << std::endl;
        outf.close();
      }
#endif
      //{
      //MatrixXcd vol = AA; // Copy Hxt at first
      //vol = vol.block<14,14>(0,0); // Pick just Hx
      //JacobiSVD<MatrixXcd, FullPivHouseholderQRPreconditioner> svd(vol);
      //VectorXd singval = svd.singularValues(); // perform SVD
      //// Zero sigma if below threshold
      //for (unsigned i = 0; i < 14; ++i)
      //  if (singval(i) < epsilon_svd)
      //    singval(i) = 0;
      //jacobian_volume = singval.prod(); // Det|Hx| at curr step
      //condition_number = singval(0)/singval(13); // c at curr step
      ////if (jacobian_volume < volthresh && condition_number > cond_thresh){ // If blows up,
      ////  std::cout << "ill-conditioned solution\n";                       // goes to the next
      ////  break;
      ////}
      //}
      memoize_Hxt(Hxt);/*, ycHxt);*/
      numerics::lsolve(AA, dxi);

      // dx3 ----
      v::multiply_scalar_to_self(dxi, one_half_dt);
      v::copy(x0t0, xt);
      v::add_to_self(xt, dxi);
      v::multiply_scalar_to_self(dxi, 4.);
      v::add_to_self(dx4, dxi);
      evaluate_Hxt(xt, params, Hxt);
#if 0
      // Gabriel Daving each preditor dx3
      if(sol_n == sol_min) {
        Matrix<std::complex<double>, 14, 14> vol;
        for (unsigned i = 0; i < 14; ++i) 
          for (unsigned j = 0; j < 14; ++j)
            vol(i,j) = AA(i,j);
        std::ofstream outf("Hxt_matrix_output_at_step_0_dx3.txt");
        outf << vol << std::endl;
        outf.close();
      }
#endif
      //{
      //MatrixXcd vol = AA; // Copy Hxt at first
      //vol = vol.block<14,14>(0,0); // Pick just Hx
      //JacobiSVD<MatrixXcd, FullPivHouseholderQRPreconditioner> svd(vol);
      //VectorXd singval = svd.singularValues(); // perform SVD
      //// Zero sigma if below threshold
      //for (unsigned i = 0; i < 14; ++i)
      //  if (singval(i) < epsilon_svd)
      //    singval(i) = 0;
      //jacobian_volume = singval.prod(); // Det|Hx| at curr step
      //condition_number = singval(0)/singval(13); // c at curr step
      ////if (jacobian_volume < volthresh && condition_number > cond_thresh){ // If blows up,
      ////  std::cout << "ill-conditioned solution\n";                       // goes to the next
      ////  break;
      ////}
      //}
      memoize_Hxt(Hxt);/*, ycHxt);*/
      numerics::lsolve(AA, dxi);

      // dx4 ----
      v::multiply_scalar_to_self(dxi, *dt);
      v::fcopy(x0t0, xt);
      v::add_to_self(xt, dxi);
      v::multiply_scalar_to_self(dxi, 2.);
      v::add_to_self(dx4, dxi);
      *t = *t0 + *dt;               // t0+dt
      evaluate_Hxt(xt, params, Hxt);
#if 0
      // Gabriel Daving each preditor dx4
      if(sol_n == sol_min) {
        Matrix<std::complex<double>, 14, 14> vol;
        for (unsigned i = 0; i < 14; ++i) 
          for (unsigned j = 0; j < 14; ++j)
            vol(i,j) = AA(i,j);
        std::ofstream outf("Hxt_matrix_output_at_step_0_dx4.txt");
        outf << vol << std::endl;
        outf.close();
      }
#endif
      {
      MatrixXcd vol = AA; // Copy Hxt at first
      vol = vol.block<14,14>(0,0); // Pick just Hx 
      JacobiSVD<MatrixXcd, FullPivHouseholderQRPreconditioner> svd(vol);
      VectorXd singval = svd.singularValues(); // perform SVD
      // Zero sigma if below threshold
      for (unsigned i = 0; i < 14; ++i)
        if (singval(i) < epsilon_svd)
          singval(i) = 0;
      jacobian_volume = singval.prod(); // Det|Hx| at curr step
      condition_number = singval(0)/singval(13); // c at curr step
      //if (jacobian_volume < volthresh && condition_number > cond_thresh){ // If blows up,
      //  std::cout << "ill-conditioned solution\n";                       // goes to the next
      //  break;
      //}
      }
      memoize_Hxt(Hxt);/*, ycHxt);*/
      numerics::lsolve(AA, dxi);
      v::multiply_scalar_to_self(dxi, *dt);
      v::add_to_self(dx4, dxi);
      v::multiply_scalar_to_self(dx4, 1./6.);

      // "dx1" = .5*dx1*dt, "dx2" = .5*dx2*dt, "dx3" = dx3*dt. Eigen vectorizes this:
      // dx4_eigen = (dx4_eigen* *dt + dx1_eigen*2 + dx2_eigen*4 + dx3_eigen*2)*(1./6.);
      
      LLOG("Solution starting point " << std::endl);
      LLOG("\t x0\n");
      pprint(x0t0, f::nve, true);
      LLOG("\t t0 " << *t0 << std::endl);
      
      // make prediction
      v::fcopy(x0t0, x1t1);
      v::fadd_to_self((double *)x1t1, (double *)dxdt);

      LLOG("Prediction" << std::endl);
      LLOG("\t x1\n");
      pprint(x1t1, f::nve, true);
      LLOG("\t t1 " << *t0 << std::endl);

      /// CORRECTOR ------------------------------------------------------------
      char n_corr_steps = 0;
      bool is_successful;
      //if (t_s->num_steps ==0)
      //evaluate_HxH_constants_all_sols(x1t1, params, ycHxH);
      //evaluate_HxH_constants(x1t1, params, ycHxH);
      /* {
            static std::mutex lock;
            const std::lock_guard<std::mutex> guard(lock);
            if ( t_s->num_steps > 1)
              for (unsigned i=0; i < 13; ++i) {
                F err = std::norm(ycHxH[i]-previous[i]);
                if (err > 1e-15) {
                  LLOG("Different " << i << " -------------------------------------------- " << err << std::endl);
                  LLOG("\tnow: " << ycHxH[i] << " previous: " << previous[i] << std::endl);
                }
              }
            for (unsigned i=0; i < 13; ++i)
              previous[i] = ycHxH[i];
         } */
      do {
        ++n_corr_steps;
        evaluate_HxH(x1t1, params, HxH);
#if 0
      // Gabriel Saving corrector first step
      if(n_corr_steps-1 ==0) {
        Matrix<std::complex<double>, 14, 14> vol;
        for (unsigned i = 0; i < 14; ++i) 
          for (unsigned j = 0; j < 14; ++j)
            vol(i,j) = AA(i,j);
        std::ofstream outf("HxH_matrix_output_at_step_0.txt");
        outf << vol << std::endl;
        outf.close();
      }
#endif
      {
        // Gabriel(12-11-2025): Create Jacobian and condition number restriction:
        // Standalone det|Hx| does not say too much since det|Hx| = prod(singularValues) and we can have a singular value many on 1e-4, but when combined with the
        // condition number I can have a clear picture since it depends solely on
        // s_0 and s_13 (c = s_0/s_13) where if S_13 is too low, the condition
        // number will skyrocket being ill-conditioned.   
        MatrixXcd vol = AA; // Copy Hxt at first
        vol = vol.block<14,14>(0,0); // Pick just Hx 
        JacobiSVD<MatrixXcd, FullPivHouseholderQRPreconditioner> svd(vol);
        VectorXd singval = svd.singularValues(); // perform SVD
        // Zero sigma if below threshold
        //for (unsigned i = 0; i < 14; ++i)
        //  if (singval(i) < epsilon_svd)
        //    singval(i) = 0;
        jacobian_volume = singval.prod(); // |Det(Hx)| at curr step
        jv = vol.determinant(); // Det|Hx| at curr step. jv = jacobian volume
        condition_number = singval(0)/singval(13); // c at curr step
      //if (jacobian_volume < volthresh && condition_number > cond_thresh){ // If blows up,
      //  std::cout << "ill-conditioned solution\n";                       // goes to the next
      //  break;
      //}
      }
        memoize_HxH(HxH);//, ycHxH);
        numerics::lsolve(AA, dx);
        v::add_to_self(x1t1, dx);
        is_successful = v::norm2(dx) < s.epsilon2_ * v::norm2(x1t1); // |dx|^2/|x1|^2 < eps2
      } while (likely(!is_successful && n_corr_steps < s.max_corr_steps_));
      
      if (unlikely(!is_successful)) { // predictor failure
        predictor_successes = 0;
        *dt *= s.dt_decrease_factor_;
        if (unlikely(*dt < s.min_dt_)) t_s->status = MIN_STEP_FAILED; // slight difference to SLP-imp.hpp:612
      } else { // predictor success
        ++predictor_successes;
        // std::swap(x1t1,x0t0);
        // x0 = x0t0; t0 = (F *) (x0t0 + f::nve); xt = x1t1;
        v::fcopy(x1t1, x0t0);
        if (unlikely(predictor_successes >= s.num_successes_before_increase_)) {
          predictor_successes = 0;
          *dt *= s.dt_increase_factor_;
        }
      }
      if (unlikely(v::norm2(x0) > s.infinity_threshold2_))
        t_s->status = INFINITY_FAILED;
      //\Gabriel: Added raw analytical data to enhance minus
      //\Local test on Chicago using worst case scenario indicates that
      //condition number --> inf if sigma_min --> 0
      //Gabriel (12-16-2025): Added breakpoint
      //\Basically if the current iteration leads to a singularity, toss it out.
      //\Why OR instead of AND?
      //\Using AND assumes an unhinged scenario which the iteration is so
      //\ill-conditioned that the only option is spit into the trash. The
      //\average case is trickier because you can have a "well behaved" volume
      //\and sigma_min numerically zero skyrocketing the condition number or in
      //\contrast sigma_max and sigma_min near in orders of magnetude but
      //\every singular value is near zero.
      if (jacobian_volume < volthresh && condition_number > cond_thresh)
        break;
      t_s->abs_det_Hx[t_s->num_steps] = jacobian_volume;
      t_s->det_Hx[t_s->num_steps] = jv;
      t_s->condition_number_Hx[t_s->num_steps] = condition_number;
      t_s->time[t_s->num_steps] = *t0;
      //\Gabriel: before my additions only had this!
      ++t_s->num_steps;
    } // while (t loop)
    memcpy(t_s, x0t0, (f::nve*2+1)*sizeof(F));
    if (t_s->status == PROCESSING) t_s->status = REGULAR;

    LLOG("Solution status " << (int)t_s->status << std::endl);
    ++t_s; s_s += f::nve;

  } // outer solution loop
}

// I/O Base functions ----------------------------------------------------------

// RC: same format as cameras_gt_ and synthcurves dataset
// QT: same format as solution_shape
template <problem P, typename F>
inline void 
minus_io_14a<P, F>::
RC_to_QT_format(const F rc[pp::nviews][4][3], F qt[M::nve])
{
  typedef minus_util<F> u;
  F q0[4], q1[4], q2[4];

  u::rotm2quat((F *) rc[0], q0);
  u::rotm2quat((F *) rc[1], q1);
  u::rotm2quat((F *) rc[2], q2);

  // gt = q1 * conj(q0);
  // gt + 4 = q2 * conj(q0);
  u::dquat(q1, q0, qt);
  u::dquat(q2, q0, qt + 4);

  // gt + 8 = q1*(c0-c1)*q1.conj();
  // gt + 8 = quat_transform(q1,c0-c1);
  // gt + 8 + 3 = quat_transform(q2,c0-c2);
  F dc[3];
  dc[0] = rc[0][3][0] - rc[1][3][0];
  dc[1] = rc[0][3][1] - rc[1][3][1];
  dc[2] = rc[0][3][2] - rc[1][3][2];
  u::quat_transform(q1,dc, qt + 8);

  dc[0] = rc[0][3][0] - rc[2][3][0];
  dc[1] = rc[0][3][1] - rc[2][3][1];
  dc[2] = rc[0][3][2] - rc[2][3][2];
  u::quat_transform(q2,dc, qt + 8 + 3);
}

// Returns all real solutions
// The real_solutions array is fixed in size to NSOLS which is the max
// number of solutions, which perfectly fits in memory. The caller must pass an
// array with that minimum.
template <problem P, typename F>
inline void 
minus_io<P, F>::
all_real_solutions(typename M::solution raw_solutions[M::nsols], F real_solutions[M::nsols][M::nve], 
                   unsigned id_sols[M::nsols], unsigned *nsols_real)
{
  typedef minus_array<M::nve,F> v;
  *nsols_real = 0;
  id_sols[*nsols_real] = 0;
  for (unsigned sol=0; sol < M::nsols; ++sol) {
    if (raw_solutions[sol].status == M::REGULAR && 
        v::get_real(raw_solutions[sol].x, real_solutions[id_sols[*nsols_real]]))
      id_sols[(*nsols_real)++] = sol;
  }
}

template <problem P, typename F>
inline void 
minus_io<P, F>::
all_regular_solutions(typename M::solution raw_solutions[M::nsols], C<F> regular_solutions[M::nsols][M::nve], 
                   unsigned id_sols[M::nsols], unsigned *nsols_regular)
{
  typedef minus_array<M::nve,F> v;
  *nsols_regular = 0;
  id_sols[*nsols_regular] = 0;
  for (unsigned sol=0; sol < M::nsols; ++sol) {
    if (raw_solutions[sol].status == M::REGULAR) {
      v::copy(raw_solutions[sol].x, regular_solutions[id_sols[*nsols_regular]]);
      id_sols[(*nsols_regular)++] = sol;
    }
  }
}

//
// Performs tests to see if there are potentially valid solutions,
// without making use of ground truth using generic tests, e.g., no real roots 
//
// This is the generic implementation. It may be specialized for each 
// problem tag
// 
template <problem P, typename F>
inline bool 
minus_io_common<P, F>::
has_valid_solutions(const typename M::solution solutions[M::nsols])
{
  typedef minus_array<M::nve,F> v;
  F real_solution[M::nve];
  for (unsigned sol = 0; sol < M::nsols; ++sol) 
    if (solutions[sol].status == M::REGULAR && v::get_real(solutions[sol].x, real_solution))
      return true;
  return false;
}

//
// Searches for probe_solution among solutions directly.
// 
// A match will occur if every coordinte is within tolerance of the probe,
// independently if it is real or not. 
//
// Multiple matches are ignored. If you need to get the minimum value, see the
// probe_solutions implementation
// 
// This default implementation is exact, meaning:the exact numerical value is
// sought. No normalizations to equivalent standard representations (eg,
// homogeneous representative) are performed.
// 
// This is the generic default implementation. It may be specialized for each
// problem tag to compare e.g. groups of variables as rotations
// 
//
template <problem P, typename F>
inline bool
minus_io<P, F>::
probe_solutions(
    const typename M::solution solutions[M::nsols], 
    C<F> probe_solution[M::nve])
{
  static constexpr F eps = 1e-3;
  for (unsigned s = 0; s < M::nsols; ++s)  {
    bool possible_match = true;
    for (unsigned v = 0; v < M::nve; ++v) {
      if (std::abs(solutions[s].x[v] - probe_solution[v]) > eps) {
        possible_match = false;
        break;
      }
    }
    if (possible_match)
      return true;
  }
  return false;
}

//
// Searches for probe_solution among solutions.
// 
// This default implementation is exact, meaning: the exact numerical value is
// sought. No normalizations to equivalent standard representations (eg,
// homogeneous representative) are performed.
//
// A match will occur if the solution is real and every coordinte is within tolerance of the probe. 
//
// _all_ means: In case of multiple matches, the solution with minimum error is returned.
// 
// Safe: this guarantees a sensibly similar match numerically, but may not be
// the fastest way to do it for your application. We coded it as efficiently as
// possible, but it is likely overkill semantically. It does not affect the main
// engine, that is why it is in a separate I/O class.
// 
// This is the generic default implementation. It may be specialized for each problem
// tag to compare e.g. groups of variables as rotations
// 
template <problem P, typename F>
inline bool
minus_io<P, F>::
probe_all_solutions(
    const typename M::solution solutions[M::nsols], 
    F probe_solution[M::nve],
    unsigned *solution_index)
{
  typedef minus_array<M::nve,F> v;
  static constexpr F eps = 1e-3;
  F real_solution[M::nve];
  bool already_found = false;
  F error, min_error;
  // Try to find the solution with minimum error
  for (unsigned sol = 0; sol < M::nsols; ++sol) {
    if (!v::get_real(solutions[sol].x, real_solution))
      continue;

    bool possible_match = true;
    error = 0;
    for (unsigned k=0; k < M::nve; ++k) {
      // the testing in this function is strict, no RMS but strictly each
      // coordinate within tolerance, even though the tolerance itself 
      // might not be strict
      F derror = std::abs(real_solution[k] - probe_solution[k]);
      if (derror > eps) {
        possible_match = false;
        break;
      }
      error += derror;
    }
    
    if (possible_match) {
      if (already_found) {
        LLOG("DEBUG: Found another similar solution at " << sol << std::endl);
        LLOG("DEBUG: Error: " << error << std::endl);
        if (error < min_error) {
          min_error = error;
          *solution_index = sol;
        }
      } else { // not already found
        LLOG("DEBUG: Found a solution at " << sol << std::endl);
        LLOG("DEBUG: Error: " << error << std::endl);
        min_error = error;
        *solution_index = sol;
      }
      already_found = true;
    } else { // not a possible solution depsite being real
      LLOG("DEBUG: Found a real solution but it is not close to ground truth: (sol,isvalid)" << sol << "," << solutions[sol].status << std::endl);
#if 0
      LLOG("\tErrors: \n");
      for (unsigned s=0; s < data::n_gt_sols_; ++s) {
        LLOG("Solution id " << s << ", errors as pairs (variable id, error): " << std::endl);
        for (unsigned v=0; v < M::nve; ++v) {
          LLOG("\t" << v << "\t" << std::abs(solutions[data::gt_sols_id_[s]].x[v] - data::gt_sols_[s][v]) << std::endl);
        }
      }
#endif
    }
  }
  
  if (!already_found)
    return false;
      
  return true;
}

//
// returns cameras[0:nsols_final][2][4][3]
//
// where the camera matrix P^t = [R|T]^t is cameras[sol_number][view_id][:][:]
// where view_id is 0 or 1 for second and third camera relative to the first,
// resp.
//
// This design is for cache speed. Translation in the camera matrix is stored
// such that its coordinates are memory contiguous.
// 
// The cameras array is fixed in size to NSOLS which is the max
// number of solutions, which perfectly fits in memory. The caller must pass an
// array with that minimum.
template <problem P, typename F>
inline void 
minus_io_14a<P, F>::
all_solutions2cams(solution raw_solutions[M::nsols], F cameras[M::nsols][2][4][3], 
                   unsigned id_sols[M::nsols], unsigned *nsols_final)
{
  typedef minus_array<M::nve,F> v;
  *nsols_final = 0;
  for (unsigned sol=0; sol < M::nsols; ++sol) {
    F real_solution[M::nve];
    if (raw_solutions[sol].status == M::REGULAR && v::get_real(raw_solutions[sol].x, real_solution)) {
      id_sols[(*nsols_final)++] = sol;
      // build cams by using quat2rotm
      solution2cams(real_solution, (F (*)[4][3] ) (cameras + sol));
    }
  }
}

// The camera parameter is cameras[img] which is a [4][3] array,
// where the first 3x3 block is R, and the 4th row is T. img is img 0 or 1,
// for 2nd and 3rd cams relative to 1st, resp.
// 
template <problem P, typename F>
inline bool 
minus_io_14a<P, F>::
probe_solutions(const typename M::solution solutions[M::nsols], solution_shape *probe_cameras,
    unsigned *solution_index)
{
  typedef minus_array<M::nve,F> v; typedef minus_util<F> u;
  static constexpr F eps = 1e-3;
  unsigned &sol=*solution_index;
  F real_solution[M::nve];
  for (sol = 0; sol < M::nsols; ++sol) 
    if (v::get_real(solutions[sol].x, real_solution)) {
      u::normalize_quat(real_solution);
      if (u::rotation_error(real_solution, probe_cameras->q01) < eps)
        return true;
    }
  return false;
}

// #undef NDEBUG

//#ifndef NDEBUG
//#include "debug-util.h"
//#endif

// like probe_solutions but tests all M::nsols in case more than one is close to
// the probe. Use this for debugging / investigation
template <problem P, typename F>
inline bool 
minus_io_14a<P, F>::
probe_all_solutions(const typename M::solution solutions[M::nsols], solution_shape *probe_cameras,
    unsigned *solution_index)
{
  typedef minus_array<M::nve,F> v; typedef minus_util<F> u;
  static constexpr F eps = 1e-3;
  F real_solution[M::nve];
  bool found = false;
  F min_rerror;
  for (unsigned sol = 0; sol < M::nsols; ++sol)  {
    if (!v::get_real(solutions[sol].x, real_solution))
      continue;
    u::normalize_quat(real_solution);
    F rerror = u::rotation_error(real_solution, probe_cameras->q01);
    if (rerror < eps) {
      if (found) {
        LLOG("Found another similar solution at " << sol << std::endl);
        LLOG("Error: " << rerror << std::endl);
        if (rerror < min_rerror) {
          min_rerror = rerror;
          *solution_index = sol;
        }
        
      } else {
        LLOG("Found a solution at " << sol << std::endl);
        LLOG("Error: " << rerror << std::endl);
        min_rerror = rerror;
        *solution_index = sol;
      }
      found = true;
    } else {
        LLOG("Solution is real but not close (sol,isvalid)" << sol << "," << solutions[sol].status << std::endl);
    }
  }

  if (!found)
    return false;

  // check the remaining parts of the solutions also match, not just rot
  v::get_real(solutions[*solution_index].x, real_solution);
  u::normalize_quat(real_solution+4);
  F rerror = u::rotation_error(real_solution+4, probe_cameras->q02);
  if (rerror < eps) {
    LLOG("probe: Rotation 02 also match\n");
    found = true;
  } else
    found = false;
  
  if (!found)
    return false;

  solution_shape *s = (solution_shape *) real_solution;

  F scale = std::sqrt(minus_3d<F>::dot(s->t01, s->t01));
  F scale_probe = std::sqrt(minus_3d<F>::dot(probe_cameras->t01, probe_cameras->t01));
  
  { // t01
  s->t01[0] /= scale; s->t01[1] /= scale; s->t01[2] /= scale;
  F dt[3];
  dt[0] = s->t01[0] - probe_cameras->t01[0]/scale_probe;
  dt[1] = s->t01[1] - probe_cameras->t01[1]/scale_probe;
  dt[2] = s->t01[2] - probe_cameras->t01[2]/scale_probe;
  
  if (minus_3d<F>::dot(dt, dt) < eps*eps) {
    LLOG("probe: translation 01 also match\n");
    found = true;
  } else {
    dt[0] = s->t01[0] + probe_cameras->t01[0]/scale_probe;
    dt[1] = s->t01[1] + probe_cameras->t01[1]/scale_probe;
    dt[2] = s->t01[2] + probe_cameras->t01[2]/scale_probe;
    if (minus_3d<F>::dot(dt, dt) < eps*eps) {
      LLOG("probe: translation 01 also match\n");
      found = true;
    } else {
      found = false;
      LLOG("probe: translation 01 DO NOT match\n");
    }
  }
  }
  
  if (!found)
    return false;
  
  { // t02
  s->t02[0] /= scale; s->t02[1] /= scale; s->t02[2] /= scale;
  F dt[3];
  dt[0] = s->t02[0] - probe_cameras->t02[0]/scale_probe;
  dt[1] = s->t02[1] - probe_cameras->t02[1]/scale_probe;
  dt[2] = s->t02[2] - probe_cameras->t02[2]/scale_probe;
  
  if (minus_3d<F>::dot(dt, dt) < eps*eps) {
    LLOG("probe: translation 02 also match\n");
    found = true;
  } else {
    //    LLOG("dt fail atttempt 1 " << std::endl);
    //    pprint(dt,3);
    //    LLOG("t02 fail atttempt 1 " << std::endl);
    //    pprint(s->t02,3);
    //    LLOG("probe t02 fail atttempt 1 " << std::endl);
    // pprint(probe_cameras->t02,3);
    
    dt[0] = s->t02[0] + probe_cameras->t02[0]/scale_probe;
    dt[1] = s->t02[1] + probe_cameras->t02[1]/scale_probe;
    dt[2] = s->t02[2] + probe_cameras->t02[2]/scale_probe;
    if (minus_3d<F>::dot(dt, dt) < eps*eps) {
      LLOG("probe: translation 02 also match\n");
      found = true;
    } else {
      found = false;
      LLOG("probe: translation 02 DO NOT match\n");
      LLOG("dt" << std::endl);
      pprint(dt,3);
    }
  }
  }
  return found;
}

// like probe_all_solutions but both solutions and ground truth probe are in
// quaternion-translation format (solution_shape)
//
// This uses real cameras as input, such as in the output of solve_img.
// 
template <problem P, typename F>
inline bool 
minus_io_14a<P, F>::
probe_all_solutions_quat(const F solutions_cameras[M::nsols][M::nve], solution_shape *probe_cameras,
    unsigned nsols, unsigned *solution_index)
{
  LLOG("Test xxxxxxxxxxx" << std::endl);
  LLOG("Nsols" <<  nsols << std::endl);
  typedef minus_util<F> u;
  static constexpr F eps = 1e-3;
  F real_solution[M::nve];
  bool found=false;
  F min_rerror;
  for (unsigned sol = 0; sol < nsols; ++sol)  {
    memcpy(real_solution, solutions_cameras[sol], M::nve*sizeof(F));
    u::normalize_quat(real_solution);
    F rerror = u::rotation_error(real_solution, probe_cameras->q01);
    if (rerror < eps) {
      if (found == true) {
        LLOG("Found another similar solution at " << sol << std::endl);
        LLOG("Error: " << rerror << std::endl);
        if (rerror < min_rerror) {
          min_rerror = rerror;
          *solution_index = sol;
        }
        
      } else {
        LLOG("Found a solution at " << sol << std::endl);
        LLOG("Error: " << rerror << std::endl);
        min_rerror = rerror;
        *solution_index = sol;
      }
      found = true;
    } 
  }

  if (!found)
    return false;

  // check the remaining parts of the solutions also match, not just rot
  memcpy(real_solution, solutions_cameras[*solution_index], M::nve*sizeof(F));
  u::normalize_quat(real_solution+4);
  F rerror = u::rotation_error(real_solution+4, probe_cameras->q02);
  if (rerror < eps) {
    LLOG("probe: Rotation 02 also match\n");
    found = true;
  } else
    found = false;
  
  if (!found)
    return false;

  solution_shape *s = (solution_shape *) real_solution;

  F scale = std::sqrt(minus_3d<F>::dot(s->t01, s->t01));
  F scale_probe = std::sqrt(minus_3d<F>::dot(probe_cameras->t01, probe_cameras->t01));
  
  { // t01
  s->t01[0] /= scale; s->t01[1] /= scale; s->t01[2] /= scale;
  F dt[3];
  dt[0] = s->t01[0] - probe_cameras->t01[0]/scale_probe;
  dt[1] = s->t01[1] - probe_cameras->t01[1]/scale_probe;
  dt[2] = s->t01[2] - probe_cameras->t01[2]/scale_probe;
  
  if (minus_3d<F>::dot(dt, dt) < eps*eps) {
    LLOG("probe: translation 01 also match\n");
    found = true;
  } else {
    dt[0] = s->t01[0] + probe_cameras->t01[0]/scale_probe;
    dt[1] = s->t01[1] + probe_cameras->t01[1]/scale_probe;
    dt[2] = s->t01[2] + probe_cameras->t01[2]/scale_probe;
    if (minus_3d<F>::dot(dt, dt) < eps*eps) {
      LLOG("probe: translation 01 also match\n");
      found = true;
    } else {
      found = false;
      LLOG("probe: translation 01 DO NOT match\n");
    }
  }
  }
  
  if (!found)
    return false;
  
  { // t02
  s->t02[0] /= scale; s->t02[1] /= scale; s->t02[2] /= scale;
  F dt[3];
  dt[0] = s->t02[0] - probe_cameras->t02[0]/scale_probe;
  dt[1] = s->t02[1] - probe_cameras->t02[1]/scale_probe;
  dt[2] = s->t02[2] - probe_cameras->t02[2]/scale_probe;
  
  if (minus_3d<F>::dot(dt, dt) < eps*eps) {
    LLOG("probe: translation 02 also match\n");
    found = true;
  } else {
    //    LLOG("dt fail atttempt 1 " << std::endl);
    //    pprint(dt,3);
    //    LLOG("t02 fail atttempt 1 " << std::endl);
    //    pprint(s->t02,3);
    //    LLOG("probe t02 fail atttempt 1 " << std::endl);
    // pprint(probe_cameras->t02,3);
    
    dt[0] = s->t02[0] + probe_cameras->t02[0]/scale_probe;
    dt[1] = s->t02[1] + probe_cameras->t02[1]/scale_probe;
    dt[2] = s->t02[2] + probe_cameras->t02[2]/scale_probe;
    if (minus_3d<F>::dot(dt, dt) < eps*eps) {
      LLOG("probe: translation 02 also match\n");
      found = true;
    } else {
      found = false;
      LLOG("probe: translation 02 DO NOT match\n");
      // LLOG("dt" << std::endl);
      // pprint(dt,3);
    }
  }
  }
  return found;
}

template <problem P, typename F>
inline bool
minus_io_14a<P, F>::
probe_solutions(const typename M::solution solutions[M::nsols], F probe_cameras[M::nve],
    unsigned *solution_index)
{
  return probe_solutions(solutions, (solution_shape *) probe_cameras, solution_index);
}

template <problem P, typename F>
inline bool
minus_io_14a<P, F>::
probe_all_solutions(const typename M::solution solutions[M::nsols], F probe_cameras[M::nve],
    unsigned *solution_index)
{
  return probe_all_solutions(solutions, (solution_shape *) probe_cameras, solution_index);
}

template <problem P, typename F>
inline bool
minus_io_14a<P, F>::
probe_all_solutions_quat(const F solutions_cameras[M::nsols][M::nve], F probe_cameras[M::nve],
    unsigned nsols, unsigned *solution_index)
{
  return probe_all_solutions_quat(solutions_cameras, (solution_shape *) probe_cameras, nsols, solution_index);
}


// For speed, assumes input point implicitly has 3rd homog coordinate is 1
// 
template <problem P, typename F>
inline void 
minus_io_common<P,F>::
invert_intrinsics(const F K[/*3 or 2 ignoring last line*/][ncoords2d_h], const double pix_coords[][ncoords2d], double normalized_coords[][ncoords2d], unsigned npts)
{
  for (unsigned p=0; p < npts; ++p) {
    const F *px = pix_coords[p];
    F *nrm = normalized_coords[p];
    nrm[1] = (px[1]-K[1][2])/K[1][1];
    nrm[0] = (px[0] - K[0][1]*nrm[1] - K[0][2])/K[0][0];
  }
}

// For speed, assumes input point implicitly has 3rd homog coordinate is 1
// 
template <problem P, typename F>
inline void 
minus_io_common<P,F>::
invert_intrinsics_tgt(const F K[/*3 or 2 ignoring last line*/][ncoords2d_h], const double pix_tgt_coords[][ncoords2d], double normalized_tgt_coords[][ncoords2d], unsigned npts)
{
  for (unsigned p=0; p < npts; ++p) {
    const F *tp = pix_tgt_coords[p];
    F *t = normalized_tgt_coords[p];
    t[1] = tp[1]/K[1][1];
    t[0] = (tp[0] - K[0][1]*t[1])/K[0][0];
  }
}

// Not sure if really necessary.
// Seemed to be important for numerics / error scales at some point.
// Normalizes line normals to unit
template <problem P, typename F>
inline void 
minus_io_common<P,F>::
normalize_lines(F lines[][ncoords2d_h], unsigned nlines)
{
  for (unsigned l=0; l < nlines; ++l)
    normalize_line(lines[l]);
}

} // namespace minus


#include "chicago14a.hxx"      // specific implementation of chicago problem, 14a formulation
#include "linecircle2a.hxx"      // specific implementation of linecircle problem, 2a formulation
//#include "cleveland14a.hxx"      // specific implementation of cleveland 14a formulation now in PLMP
// #include <minus/phoenix10a.hxx>      // specific implementation of chicago 14a formulation
// #include "chicago6a.hxx"

#endif // minus_hxx_

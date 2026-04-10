// \author Ricardo Fabbri based on original code by Anton Leykin 
// \date February 2019-2026
// 
#include "minus-chicago.h"

int
find_ground_truth(M::solution solutions[M::nsols])
{
  std::cerr << "Usage: minus [input solutions]\n\n";
  std::cerr << "If no argument is given, 'input' is assumed stdin,\n\
  'solutions' will be output to stdout\n";
  std::cerr << "Example: \n"
               "  minus input_file solutions_file\n"
               "  minus <input_file >solutions_file\n"
               "  minus -g       # (or --profile) : performs a default solve for profiling\n"
               "  minus -i       # (or --image_data) : reads point-tangents from stdin\n"
               "  minus -h       # (or --help) : print this help message\n"
               // "  minus -r       # (or --real)  :  outputs only real solutions\n"
               "  minus -AB      # (or --two_problems) : continue between 2 given problems\n"
            <<
  R"(-i | --image_data usage:
 
  Input format (notation _view_points_coords. any number of spaces and newlines optional. can be in
  one row or one column as well). This input format assumes tangent data for
  all points, but you specify which one to use in id0 and id1 below. When
  --use_all_tangents is passed (TODO), will try to select the better conditioned / least degenerate tangents 
 
  p000 p001        # If continuing from a standard internal problem to a new problem A, this is problem A
  p010 p011        # If continuing from two problems from A to B (flag -AB), this is also problem A
  p020 p021
  
  p100 p101
  p110 p111
  p120 p121
  
  p100 p101
  p110 p111
  p120 p121
 
  t000 t001
  t010 t011
  t020 t021
  
  t100 t101
  t110 t111
  t120 t121
  
  t100 t101
  t110 t111
  t120 t121
  
  id0 id1           # id \in {0,1,2} of the point to consider the tangent
  
  K00 K01 K02       # intrinsic parameters: only these elements
   0  K11 K22

  r000 r001 r002    # GROUND TRUTH (optional) if -gt flag provided, pass the ground truth here:
  r010 r011 r012    # default camera format if synthcurves flag passed: 
  r020 r021 r022    # just like a 3x4 [R|T] but transposed to better fit row-major:
   c00  c01  c02    #         | R |
                    # P_4x3 = | - |
  r100 r101 r102    #         | C'|
  r110 r111 r112    # 
  r120 r121 r122    #  
   c10  c11  c12    #                                                                                                                   # If two problems A->B are provided (flag -AB), this is only for problem B below
                    #
  r200 r201 r202    # 
  r210 r211 r212    # 
  r220 r221 r222    #
   c20  c21  c22    # 

  p000 p001         # If two problems A->B are provided (flag -AB), this is problem B
  p010 p011
  p020 p021
  
  p100 p101
  p110 p111
  p120 p121
  
  p100 p101
  p110 p111
  p120 p121
 
  t000 t001
  t010 t011
  t020 t021
  
  t100 t101
  t110 t111
  t120 t121
  
  t100 t101
  t110 t111
  t120 t121
  
  id0 id1           # id \in {0,1,2} of the point to consider the tangent

  # One way to use this is 
  #     synthdata | minus-chicago -i
  # where synthdata is provided in minus/scripts)";
             
  exit(1);
}

bool stdio_ = true;  // by default read/write from stdio
bool ground_truth_ = false;
bool two_problems_given_ = false;
bool reading_first_point_ = true;
std::ifstream infp_;
bool image_data_ = false;
bool profile_ = false;   // run some default solves for profiling
const char *input_ = "stdin";
const char *output_ = "stdout";
M::track_settings settings_; // general homotopy settings
M::f::settings ssettings_;   // specific settings (formulation-specific)

void
print_num_steps(M::solution solutions[M::nsols])
{
  LOG("solution id x num steps:");
  unsigned sum=0;
  for (unsigned s=0; s < M::nsols; ++s) {
    LOG(s << " " << solutions[s].num_steps);
    sum += solutions[s].num_steps;
  }
  LOG("total number of steps: " << sum);
}
//\Gabriel (12-12-2025): New features Print det, condition number
//\of each solution as raw data
void print_volume(M::solution solutions[M::nsols]){
  LOG("solution id x volume:");
  std::cout<< "solution id x volume:"<< std::endl;
  unsigned sum=0;
  for (unsigned s=0; s < M::nsols; ++s) {
    for(unsigned v=0; v < solutions[s].num_steps; ++v){
      LOG(s << " " << solutions[s].det_Hx[v]);
      std::cout << s << " " << solutions[s].det_Hx[v] << std::endl;
    }
  }
}
void print_cn(M::solution solutions[M::nsols]){
  LOG("solution id x condition_number:");
  std::cout<< "solution id x condition_number:"<< std::endl;
  unsigned sum=0;
  for (unsigned s=0; s < M::nsols; ++s) {
    for(unsigned v=0; v < solutions[s].num_steps; ++v) {
      LOG(s << " " << solutions[s].condition_number_Hx[v]);
      std::cout << s << " " << solutions[s].condition_number_Hx[v] << std::endl;
    }
  }
}
void print_time(M::solution solutions[M::nsols]){
  LOG("solution id x time:");
  std::cout<< "solution id x time:"<< std::endl;
  unsigned sum=0;
  for (unsigned s=0; s < M::nsols; ++s) {
    for(unsigned v=0; v < solutions[s].num_steps; ++v) {
      LOG(s << " " << solutions[s].time[v]);
      std::cout << s << " " << solutions[s].time[v] << std::endl;
    }
  }
}
// Output solutions in ASCII matlab format
//
// ---------------------------------------------------------
// If in the future our solver is really fast, we may need Binary IO:
// complex solutions[NSOLS*NVE];
// 
// To read this output file in matlab, do:
// fid = fopen(fname,'r');
// a_raw = fread(fid,'double');
// fclose(fid);
//
// Reshape a to have proper real and imaginary parts
// a = a_raw(1:2:end) + i*a_raw(2:2:end);
// 
template <typename F=double>
static bool
mwrite(const M::solution s[M::nsols], const char *fname)
{
  bool scilab=false;
  std::string imag("+i*");
  if (scilab) imag = std::string("+%i*");
    
  std::ofstream fsols;
  std::streambuf *buf;
  
  if (stdio_) {
    buf = std::cout.rdbuf();
    std::cout << std::setprecision(20);
  // TODO(juliana) should we has_valid_solutions here? 
  io::RC_to_QT_format(data::cameras_gt_, data::cameras_gt_quat_);
  unsigned sol_id;
  if (io::probe_all_solutions(solutions, data::cameras_gt_quat_, &sol_id)) {
    LOG("found solution at index: " << sol_id);
    LOG("number of iterations of solution: " << solutions[sol_id].num_steps);
    if (solutions[sol_id].status != M::REGULAR)
      LOG("PROBLEM found ground truth but it is not REGULAR: " << sol_id);
  } else {
    LOG("\033[1;91mFAIL:\e[m  ground-truth not found among solutions");
    return SOLVER_FAILURE; 
    // you can detect solver failure by checking this exit code.
    // if you use shell, see:
    // https://www.thegeekstuff.com/2010/03/bash-shell-exit-status
  }
  return 0;
}

void 
run_solver(M::solution solutions[M::nsols])
{
  LOG("\033[0;33mUsing 4 threads by default\e[m\n");
  #ifdef M_VERBOSE
  if (two_problems_given_)
    std::cerr 
      << "LOG \033[0;33mContinuing between two problems A -> B by internal 0->A then A->B\e[m\n" 
      << "LOG \033[0;33mStarting path tracker from random initial solution to first problem 0->A\e[m\n" 
      << std::endl;
  else
    std::cerr << "LOG \033[0;33mStarting path tracker from random initial solution to given problem\e[m\n" << std::endl;
  #endif 
  std::thread t[4];
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  //  unsigned retval = 
  //  ptrack(&MINUS_DEFAULT, start_sols_, params_, solutions);
  {
    t[0] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 0, 78);
    t[1] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 78, 78*2);
    t[2] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 78*2, 78*3);
    t[3] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 78*3, 78*4);
    t[0].join(); t[1].join(); t[2].join(); t[3].join();
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(t2 - t1).count();
  #ifdef M_VERBOSE
  print_num_steps(solutions);
  std::cerr << "LOG \033[1;32mTime of solver: " << duration << "ms\e[m" << std::endl;
  #endif
}


// exact means the exact numerical value, without normalizing, for trying to
// reproduce a specific numeric behavior
void
probe_solutions_with_gt_exact(M::solution solutions[M::nsols])
{
  // override default data::probe_solutions(sols);
  // 
  // compare solutions to certain values from M2
  // two random entries
  if (std::abs(solutions[1].x[1] - complex(-.25177177692982444e1, -.84845195030295639)) <= tol &&
      std::abs(solutions[M::nsols-2].x[2] - complex(.7318330016224166, .10129116603501138)) <= tol)
    std::cerr << "LOG solutions look OK\n";
  else  {
    std::cerr << "LOG \033[1;91merror:\e[m solutions dont match m2. Errors: ";
    std::cerr << std::abs(solutions[1].x[2] - complex(-.25177177692982444e1, -.84845195030295639)) << ", "
        << std::abs(solutions[M::nsols-2].x[2] - complex(.7318330016224166, .10129116603501138)) << std::endl;
  }
}

// Experimental
// TODO: move out to algo/
int
run_solver_AB(M::solution solutions[M::nsols], std::istream *inp = &std::cin)
{
  // First, lets make sure problem A was properly solved
  if (!io::has_valid_solutions(solutions)) { // if no ground-truth is provided, it will return error
    LOG("\033[1;91mFAIL:\e[m  no valid solutions in problem A");
    return SOLVER_FAILURE;                    // if it can detect that the solver failed by generic tests
  }
  // 
  // Continue between two problems A and B by continuing from an internal
  // problem R to A (to discover all solutions of A), then from A to B.
  //
  // format solutions (of A) to be similar to data::start_sols_
  complex sols_A_matrix[M::nsols][M::nve];
  io::solutions_struct2vector(solutions, sols_A_matrix);
  const complex * const sols_A = (complex *) sols_A_matrix;
  
  // reset solutions
  static const M::solution s0;
  for (unsigned s=0; s < M::nsols; ++s)
    solutions[s] = s0;

  // generate homotopy params_ -----------------------------------------------
  //
  // At this point:
  // params_start_target_ = [ P0gammified, PAgammified]
  //    
  // We want
  // params_start_target_ = [ PAgammified, PBbammified]
  //
  // First we do params_start_target  = [ PAgammified, PAgammified]
  memcpy(data::params_start_target_, 
         data::params_start_target_+M::f::nparams, M::f::nparams*sizeof(complex));
  
  LOG("\033[0;33mReading second problem B\e[m\n");
  // Now read problem B & extract parameters into 2nd half of
  // params_start_target_
  if (input_data_) {  // read image pixel-based I/O parameters
    if (!iread<Float>(*inp))
      return 1;
    data::params_ = data::params_start_target_;
  } else {  // read raw I/O homotopy parameters (to be used as engine)
    std::cerr << "When continuing from A to B, non-pixel input not implemented\n";
    return 1;
  }
  
  // Homotopy-continue from A to B ---------------------------------------
  LOG("\033[0;33mUsing 4 threads by default\e[m\n");
  #ifdef M_VERBOSE
  std::cerr << "LOG \033[0;33mStarting path tracker from A to B\e[m\n" << std::endl;
  #endif 
  std::thread t[4];
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  //  unsigned retval = 
  //  ptrack(&MINUS_DEFAULT, start_sols_, params_, solutions);
  {
    t[0] = std::thread(M::track, settings_, sols_A, data::params_, solutions, 0, 78);
    t[1] = std::thread(M::track, settings_, sols_A, data::params_, solutions, 78, 78*2);
    t[2] = std::thread(M::track, settings_, sols_A, data::params_, solutions, 78*2, 78*3);
    t[3] = std::thread(M::track, settings_, sols_A, data::params_, solutions, 78*3, 78*4);
    t[0].join(); t[1].join(); t[2].join(); t[3].join();
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(t2 - t1).count();
  #ifdef M_VERBOSE
  print_num_steps(solutions);
  std::cerr << "LOG \033[1;32mTime of solver A -> B: " << duration << "ms\e[m" << std::endl;
  #endif
  return 0;
}

// Simplest possible command to compute the Chicago problem
// for estimating calibrated trifocal geometry from points and lines at points
//
// This is to be kept very simple C with only minimal C++ with Templates.
// If you want to complicate this, please create another executable.
// 
int
main(int argc, char **argv)
{
  std::istream *inp = &std::cin;
  cmd c;
  
  process_args(c, argc, argv);
  print_all_settings(settings_, ssettings_);

  if (!profile_) { // Read files: either stdio or physical
    c.init_input(c.input_, inp);
    if (input_data_) {          // Read target problem data, which is then converted to
      if (!iread<Float>(*inp))  // the internally used problem parameters
        return 1;
      data::params_ = data::params_start_target_;
    } else {  // Read raw start+target homotopy parameters, possibly randomized. (To be used as engine)
      if (!c.mread(*inp))  // Reads into global params_
        return 1;
    }
  } // else, profile: the homotopy data is already hardcoded in data::params_
  
  alignas(64) static M::solution solutions[M::nsols];
  {
    LOG("\033[0;33mUsing 4 threads by default\e[m\n");
    #ifdef M_VERBOSE
    if (two_problems_given_)
      std::cerr 
        << "LOG \033[0;33mContinuing between two problems A -> B by internal 0->A then A->B\e[m\n" 
        << "LOG \033[0;33mStarting path tracker from random initial solution to first problem 0->A\e[m\n" 
        << std::endl;
    else
      std::cerr << "LOG \033[0;33mStarting path tracker from random initial solution to given problem\e[m\n" << std::endl;
    #endif 
    std::thread t[4];
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    //  unsigned retval = 
    //  ptrack(&MINUS_DEFAULT, start_sols_, params_, solutions);
    {
      t[0] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 0, 78);
      t[1] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 78, 78*2);
      t[2] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 78*2, 78*3);
      t[3] = std::thread(M::track, settings_, data::start_sols_, data::params_, solutions, 78*3, 78*4);
      t[0].join(); t[1].join(); t[2].join(); t[3].join();
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    #ifdef M_VERBOSE
    print_num_steps(solutions);
    print_volume(solutions);
    print_cn(solutions);
    print_time(solutions);
    std::cerr << "LOG \033[1;32mTime of solver: " << duration << "ms\e[m" << std::endl;
    #endif
  }

  
  run_solver(solutions);
  
  if (two_problems_given_) {
    int stat = run_solver_AB(solutions, inp);
    if (!stat) return stat;
  }
  
  if (profile_)
    probe_solutions_with_gt_exact(solutions);
  
  if (!c.mwrite(solutions, c.output_)) return 2;

  if (ground_truth_ || profile_)
    return find_ground_truth(solutions);
  else if (!io::has_valid_solutions(solutions)) {      // if no ground-truth is provided, it will return error if
    LOG("\033[1;91mFAIL:\e[m  no valid real solutions"); // it can detect that the solver failed by generic tests
    return SOLVER_FAILURE;                               // without using ground-truth, e.g., no real roots
  }                                                      // or problem-specific inequalities
  return 0;
}

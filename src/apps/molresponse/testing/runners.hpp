//
// Created by adrianhurtado on 2/11/22.
//

#ifndef MADNESS_RUNNERS_HPP
#define MADNESS_RUNNERS_HPP

#include "ExcitedResponse.hpp"
#include "FrequencyResponse.hpp"
#include "ResponseExceptions.hpp"
#include "TDDFT.h"
#include "madness/chem/SCF.h"
#include "madness/tensor/tensor_json.hpp"
#include "madness/world/worldmem.h"
#include "response_data_base.hpp"
#include "response_functions.h"
#include "sstream"
#include "string"
#include "timer.h"
#include "write_test_input.h"
#include "x_space.h"

auto split(const std::string &s, char delim) -> vector<std::string> {
    vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) { result.push_back(item); }

    return result;
}

auto addPath(const path &root, const std::string &branch) -> path {

    path p_branch = root;
    p_branch += branch;
    return p_branch;
}

struct runSchema {
    path root;               // root directory
    path molecule_path;      // molecule directory
    path xc_path;            // create xc path
    path freq_json;          // path to freq_json
    path dalton_dipole_json; // path to dalton to dipole json
    path dalton_excited_json;// path to dalton excited json
    ResponseDataBase rdb;

    explicit runSchema(const std::string &xc) {

        root = std::filesystem::current_path();//="/"+molecule_name;
        molecule_path = root;
        molecule_path += "/molecules";

        xc_path = addPath(root, "/" + xc);
        if (std::filesystem::exists(xc_path)) {
            std::cout << "XC Directory Exists" << std::endl;
        } else {
            std::cout << "Creating XC directory" << std::endl;
            std::filesystem::create_directory(xc_path);
        }

        // Get the database where the calculation will be run from
        freq_json = addPath(molecule_path, "/frequency.json");
        dalton_excited_json = addPath(molecule_path, "/dalton-excited.json");
        dalton_dipole_json = addPath(molecule_path, "/dalton-dipole.json");

        rdb = ResponseDataBase();


        if (std::filesystem::exists(freq_json)) {
            std::ifstream ifs(freq_json);
            std::cout << "Trying to read frequency.json" << std::endl;
            json j_read;
            ifs >> j_read;
            std::cout << "READ IT" << std::endl;
            rdb.set_data(j_read);

        } else {
            std::cout << "did not find frequency.json" << std::endl;
        }
        print();
    }

    void print() const {

        ::print("------------Database Runner---------------");
        ::print("Root: ", root);
        ::print("Molecule Directory: ", molecule_path);
        ::print("XC Path: ", xc_path);
        ::print("Freq Json Path: ", freq_json);
        ::print("Dalton Dipole Json Path: ", dalton_dipole_json);
        ::print("Dalton Excited Json Path: ", dalton_excited_json);
    }
};

struct moldftSchema {

    path moldft_path;
    path moldft_json_path;
    json moldft_json;

    path moldft_restart;
    path calc_info_json_path;
    json calc_info_json;
    path mol_path;
    std::string mol_name;
    std::string xc;

    moldftSchema(const std::string &molecule_name, const std::string &m_xc, const runSchema &schema)
        : mol_name(molecule_name),
          xc(m_xc) {

        moldft_path = addPath(schema.xc_path, '/' + mol_name);
        moldft_restart = addPath(moldft_path, "/moldft.restartdata.00000");
        calc_info_json_path = addPath(moldft_path, "/moldft.calc_info.json");
        mol_path = addPath(schema.molecule_path, "/" + mol_name + ".mol");

        moldft_json_path = addPath(schema.molecule_path, "/moldft.json");

        if (std::filesystem::exists(moldft_json_path)) {
            std::ifstream ifs(moldft_json_path);
            // read results into json
            ifs >> moldft_json;
            // Here are the current answers... check to see if th
            std::cout << "Here are the current answers for" << molecule_name
                      << " check to see if they need to be updated please!" << std::endl;
            cout << moldft_path;
        } else {
            std::cout << " We do not have moldft answers so please run and save the "
                         "results in the molecule directory"
                      << std::endl;
        }

        if (std::filesystem::exists(moldft_restart) &&
            std::filesystem::exists(calc_info_json_path)) {
            // if both exist, read the calc_info json
            std::ifstream ifs(calc_info_json_path);
            ifs >> calc_info_json;
            std::cout << "time: " << calc_info_json["time"] << std::endl;
            std::cout << "MOLDFT return energy: " << calc_info_json["return_energy"] << std::endl;
            std::cout << "MOLDFT return energy answer: " << moldft_json["return_energy"]
                      << std::endl;
        }
        print();
    }

    void print() const {
        ::print("----------------- Moldft Paths --------------------");
        ::print("moldft path :", moldft_path);
        ::print("moldft json path :", moldft_json_path);
        ::print("moldft restart path :", moldft_restart);
        ::print("molecule path  path :", mol_path);
        ::print("calc_info json path :", calc_info_json_path);
        ::print("moldft json path :", moldft_json_path);
    }
};

struct frequencySchema {

    const std::string mol_name;
    const std::string xc;
    const std::string op;

    const path moldft_path;
    vector<double> freq;

    frequencySchema(const runSchema run_schema, const moldftSchema m_schema,
                    const std::string r_operator)
        : mol_name(m_schema.mol_name),
          xc(m_schema.xc),
          op(r_operator),
          moldft_path(m_schema.moldft_path) {
        freq = run_schema.rdb.get_frequencies(mol_name, xc, op);
        print_schema();
    }

    void print_schema() {
        print("Frequency Calculation");
        print("Molecule Name: ", mol_name);
        print("Functional: ", xc);
        print("Operator: ", op);
        print("MOLDFT PATH: ", moldft_path);
        print("Frequencies : ", freq);
    }
};

/**
 * Sets the excited state data found in the response_data_base class
 * If the data is not found it will just return 4
 * @param response_data_base
 * @param molecule_path
 * @param molecule_name
 * @param xc
 * @param property
 * @return
 */
size_t set_excited_states(const ResponseDataBase &response_data_base,
                          const std::filesystem::path &molecule_path,
                          const std::string &molecule_name, const std::string &xc) {

    const std::string property = "excited-state";

    try {
        return response_data_base.get_num_states(molecule_name, xc, property);
    } catch (json::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << "did not find the frequency data for [" << molecule_name << "][" << xc << "]["
                  << property << "]\n";
        return 4;
    }
}

/**
 * generates the frequency response path using the format
 * [property]_[xc]_[1-100]
 *
 * where 1-100 corresponds a frequency of 1.100
 *
 * @param moldft_path
 * @param property
 * @param frequency
 * @param xc
 * @return
 */
std::filesystem::path generate_excited_run_path(const std::filesystem::path &moldft_path,
                                                const size_t &num_states, const std::string &xc) {
    std::string s_num_states = std::to_string(num_states);
    std::string run_name = "excited-" + s_num_states;
    // set r_params to restart true if restart file exist

    auto run_path = moldft_path;
    run_path += "/";
    run_path += std::filesystem::path(run_name);
    std::cout << run_path << endl;
    return run_path;
}
// sets the current path to the save path
/**
 * Generates the frequency save path with format
 * /excited_state/restart_[frequency_run_filename].00000
 *
 * @param excited_state restart path
 * @return
 */
std::pair<std::filesystem::path, std::string> generate_excited_save_path(
        const std::filesystem::path &excited_run_path) {

    auto save_path = std::filesystem::path(excited_run_path);
    std::string save_string = "restart_excited";
    save_path += "/";
    save_path += save_string;

    save_path += ".00000";
    return {save_path, save_string};
}

struct excitedSchema {
    std::string xc;
    size_t num_states;
    path excited_state_run_path;
    path save_path;
    std::string save_string;

    path rb_json;


    excitedSchema(const runSchema &run_schema, const moldftSchema &m_schema) : xc(m_schema.xc) {
        num_states =
                set_excited_states(run_schema.rdb, run_schema.molecule_path, m_schema.mol_name, xc);
        excited_state_run_path = generate_excited_run_path(m_schema.moldft_path, num_states, xc);
        auto [sp, s] = generate_excited_save_path(excited_state_run_path);
        save_path = sp;
        save_string = s;
        rb_json = addPath(excited_state_run_path, "/response_base.json");
    }

    void print() {

        ::print("xc: ", xc);
        ::print("num states: ", num_states);
        ::print("excited_state run_path: ", excited_state_run_path);
        ::print("save_path: ", save_path);
        ::print("save_string: ", save_string);
    }
};

/**
 * Creates the xc directory in root directory of the
 *
 * Will create the xc directory if it does not already exist. Returns the path
 * of xc directory
 *
 *
 * @param root
 * @param xc
 * @return xc_path
 */
std::filesystem::path create_xc_path_and_directory(const std::filesystem::path &root,
                                                   const std::string &xc) {

    // copy construct the  root path
    auto xc_path = std::filesystem::path(root);
    xc_path += "/";
    xc_path += std::filesystem::path(xc);
    if (std::filesystem::is_directory(xc_path)) {

        cout << "XC directory found " << xc << "\n";

    } else {// create the file
        std::filesystem::create_directory(xc_path);
        cout << "Creating XC directory for " << xc << ":\n";
    }

    return xc_path;
}

// sets the current path to the save path
/**
 * Generates the frequency save path with format
 * /frequency_run_path/restart_[frequency_run_filename].00000
 *
 * @param frequency_run_path
 * @return
 */
auto generate_frequency_save_path(const std::filesystem::path &frequency_run_path)
        -> std::pair<std::filesystem::path, std::string> {

    auto save_path = std::filesystem::path(frequency_run_path);
    auto run_name = frequency_run_path.filename();
    std::string save_string = "restart_" + run_name.string();
    save_path += "/";
    save_path += save_string;

    save_path += ".00000";
    return {save_path, save_string};
}

/**
 * generates the frequency response path using the format
 * [property]_[xc]_[1-100]
 *
 * where 1-100 corresponds a frequency of 1.100
 *
 * @param moldft_path
 * @param property
 * @param frequency
 * @param xc
 * @return
 */
auto generate_response_frequency_run_path(const std::filesystem::path &moldft_path,
                                          const std::string &property, const double &frequency,
                                          const std::string &xc) -> std::filesystem::path {
    std::string s_frequency = std::to_string(frequency);
    auto sp = s_frequency.find(".");
    s_frequency = s_frequency.replace(sp, sp, "-");
    std::string run_name = property + "_" + xc + "_" + s_frequency;
    // set r_params to restart true if restart file exist

    auto run_path = moldft_path;
    run_path += "/";
    run_path += std::filesystem::path(run_name);
    std::cout << run_path << endl;
    return run_path;
}

/**
 * Reads in current parameters and found parameters from calc_info.json and determines whether we should restart the calculation
 * @param p1
 * @param p2
 * @return
 */
auto tryMOLDFT(CalculationParameters &p1, CalculationParameters &p2) -> bool {

    // first get the last protocol

    auto proto1 = p1.get<std::vector<double>>("protocol");
    auto proto2 = p1.get<std::vector<double>>("protocol");

    std::cout << *(proto2.end() - 1) << std::endl;
    std::cout << *(proto1.end() - 1) << std::endl;

    return *(proto1.end() - 1) != *(proto2.end() - 1);
}

/**
 * Runs moldft in the path provided.  Also generates the moldft input file_name
 * in the directory provided.
 *
 * @param world
 * @param moldft_path
 * @param moldft_filename
 * @param xc
 */
void runMOLDFT(World &world, const moldftSchema &moldftSchema, bool try_run, bool restart,
               bool high_prec) {

    CalculationParameters param1;

    param1.set_user_defined_value("maxiter", 20);
    //param1.set_user_defined_value("Kain", true);

    param1.set_user_defined_value<std::string>("xc", moldftSchema.xc);
    param1.set_user_defined_value<double>("l", 200);

    if (high_prec) {
        param1.set_user_defined_value<vector<double>>("protocol", {1e-4, 1e-6, 1e-8});
        param1.set_user_defined_value<double>("dconv", 1e-6);
    } else {
        param1.set_user_defined_value<vector<double>>("protocol", {1e-4, 1e-6});
        param1.set_user_defined_value<double>("dconv", 1e-4);
    }

    param1.set_user_defined_value<std::string>("localize", "new");

    CalculationParameters param_calc;
    json calcInfo;
    if (std::filesystem::exists(moldftSchema.calc_info_json_path)) {
        std::cout<<"Reading Calc Info JSON"<<std::endl;
        std::ifstream ifs(moldftSchema.calc_info_json_path);
        ifs >> calcInfo;
        param_calc.from_json(calcInfo["parameters"]);
        print(param1.print_to_string());
        print(param_calc.print_to_string());
    }
    //If the parameters are exactly equal do not run
    // If calc info doesn't exist the param_calc will definitely be different

    // if parameters are different or if I want to run no matter what run
    // if I want to restart and if I can. restart
    print("param1 != param_calc = ", param1 != param_calc);
    if (tryMOLDFT(param1, param_calc) || try_run) {
        print("-------------Running moldft------------");
        // if params are different run and if restart exists and if im asking to restar
        if (std::filesystem::exists(moldftSchema.moldft_restart) && restart) {
            param1.set_user_defined_value<bool>("restart", true);
        }
        molresponse::write_test_input test_input(param1, "moldft.in",
                                                 moldftSchema.mol_path);// molecule HF
        commandlineparser parser;
        parser.set_keyval("input", test_input.filename());
        SCF calc(world, parser);
        calc.set_protocol<3>(world, 1e-4);
        MolecularEnergy ME(world, calc);
        // double energy=ME.value(calc.molecule.get_all_coords().flat()); // ugh!
        ME.value(calc.molecule.get_all_coords().flat());// ugh!
        ME.output_calc_info_schema();
    } else {
        print("Skipping Calculation and printing CALC INFO");
        std::cout << calcInfo;
    }
}

/**
 * Sets the response parameters for a frequency response calculation and writes
 * to file
 *
 * @param r_params
 * @param property
 * @param xc
 * @param frequency
 */
void set_excited_parameters(ResponseParameters &r_params, const std::string &xc,
                            const size_t &num_states, bool high_prec) {


    if (high_prec) {
        r_params.set_user_defined_value<vector<double>>("protocol", {1e-4, 1e-6, 1e-8});
        r_params.set_user_defined_value<double>("dconv", 1e-6);
    } else {
        r_params.set_user_defined_value<vector<double>>("protocol", {1e-4, 1e-6});
        r_params.set_user_defined_value<double>("dconv", 1e-4);
    }
    //r_params.set_user_defined_value("archive", std::string("../restartdata"));
    r_params.set_user_defined_value("maxiter", size_t(15));
    r_params.set_user_defined_value("maxsub", size_t(10));
    // if its too large then bad guess is very strong
    r_params.set_user_defined_value("kain", true);
    r_params.set_user_defined_value("plot_all_orbitals", false);
    r_params.set_user_defined_value("save", true);
    r_params.set_user_defined_value("guess_xyz", false);
    r_params.set_user_defined_value("print_level", 20);
    // set xc, property, num_states,and restart
    r_params.set_user_defined_value("xc", xc);
    r_params.set_user_defined_value("excited_state", true);
    r_params.set_user_defined_value("states", num_states);
    // Here
}

/**
 * Sets the response parameters for a frequency response calculation and writes
 * to file
 *
 * @param r_params
 * @param property
 * @param xc
 * @param frequency
 */
void set_frequency_response_parameters(ResponseParameters &r_params, const std::string &property,
                                       const std::string &xc, const double &frequency,
                                       bool high_precision) {
    if (high_precision) {
        r_params.set_user_defined_value<vector<double>>("protocol", {1e-4, 1e-6, 1e-8});
        r_params.set_user_defined_value<double>("dconv", 1e-6);
    } else {
        r_params.set_user_defined_value<vector<double>>("protocol", {1e-4, 1e-6, 1e-6});
        r_params.set_user_defined_value<double>("dconv", 1e-4);
    }
    //r_params.set_user_defined_value("archive", std::string("../restartdata"));
    r_params.set_user_defined_value("maxiter", size_t(30));
    r_params.set_user_defined_value("maxsub", size_t(5));
    r_params.set_user_defined_value("kain", true);
    r_params.set_user_defined_value("omega", frequency);
    r_params.set_user_defined_value("first_order", true);
    r_params.set_user_defined_value("plot_all_orbitals", false);
    r_params.set_user_defined_value("print_level", 20);
    r_params.set_user_defined_value("save", true);
    // set xc, property, frequency,and restart
    r_params.set_user_defined_value("xc", xc);
    // Here
    if (property == "dipole") {
        r_params.set_user_defined_value("dipole", true);
    } else if (property == "nuclear") {
        r_params.set_user_defined_value("nuclear", true);
    }
}

/***
 * sets the run path based on the run type set by r_params
 * creates the run directory and sets current directory to the run data
 * returns the name of parameter file to run from
 *
 * @param parameters
 * @param frequency
 * @param moldft_path
 */
static auto set_frequency_path_and_restart(ResponseParameters &parameters,
                                           const std::string &property, const double &frequency,
                                           const std::string &xc,
                                           const std::filesystem::path &moldft_path,
                                           std::filesystem::path &restart_path, bool restart)
        -> std::filesystem::path {

    print("set_frequency_path_and_restart");
    print("restart path", restart_path);


    // change the logic create save path
    auto frequency_run_path =
            generate_response_frequency_run_path(moldft_path, property, frequency, xc);
    print("frequency run path", frequency_run_path);
    // Crea
    if (std::filesystem::is_directory(frequency_run_path)) {
        cout << "Response directory found " << std::endl;
    } else {// create the file
        std::filesystem::create_directory(frequency_run_path);
        cout << "Creating response_path directory" << std::endl;
    }

    std::filesystem::current_path(frequency_run_path);
    // Calling this function will make the current working directory the
    // frequency save path
    auto [save_path, save_string] = generate_frequency_save_path(frequency_run_path);
    print("save string", save_string);

    parameters.set_user_defined_value("save", true);
    parameters.set_user_defined_value("save_file", save_string);
    // if restart and restartfile exists go ahead and set the restart file
    if (restart) {
        if (std::filesystem::exists(save_path)) {

            parameters.set_user_defined_value("restart", true);
            parameters.set_user_defined_value("restart_file", save_string);
            print("found save directory... restarting from save_string ", save_string);
        } else if (std::filesystem::exists(restart_path)) {

            print(" restart path", restart_path);
            parameters.set_user_defined_value("restart", true);

            auto split_restart_path = split(restart_path.replace_extension("").string(), '/');

            std::string restart_file_short =
                    "../" + split_restart_path[split_restart_path.size() - 2] + "/" +
                    split_restart_path[split_restart_path.size() - 1];
            print("relative restart path", restart_file_short);
            parameters.set_user_defined_value("restart_file", restart_file_short);
            print("found restart directory... restarting from restart_path.string ",
                  restart_path.string());

        } else {
            parameters.set_user_defined_value("restart", false);
            // neither file exists therefore you need to start from fresh
        }
    } else {
        parameters.set_user_defined_value("restart", false);
    }
    return save_path;
}

/**
 *
 * @param world
 * @param filename
 * @param frequency
 * @param property
 * @param xc
 * @param moldft_path
 * @param restart_path
 * @return
 */
auto RunResponse(World &world, const std::string &filename, double frequency,
                 const std::string &property, const std::string &xc,
                 const std::filesystem::path &moldft_path, std::filesystem::path restart_path,
                 bool highPrecision) -> std::pair<std::filesystem::path, bool> {

    // Set the response parameters
    ResponseParameters r_params{};
    set_frequency_response_parameters(r_params, property, xc, frequency, highPrecision);
    auto save_path = set_frequency_path_and_restart(r_params, property, frequency, xc, moldft_path,
                                                    restart_path, true);

    if (world.rank() == 0) { molresponse::write_response_input(r_params, filename); }
    // if rbase exists and converged I just return save path and true
    if (std::filesystem::exists("response_base.json")) {
        std::ifstream ifs("response_base.json");
        json response_base;
        ifs >> response_base;
        if (response_base["converged"]) { return {save_path, true}; }
    }
    auto calc_params = initialize_calc_params(world, std::string(filename));
    RHS_Generator rhs_generator;
    if (property == "dipole") {
        rhs_generator = dipole_generator;
    } else {
        rhs_generator = nuclear_generator;
    }
    FrequencyResponse calc(world, calc_params, frequency, rhs_generator);
    if (world.rank() == 0) {
        print("\n\n");
        print(" MADNESS Time-Dependent Density Functional Theory Response "
              "Program");
        print(" ----------------------------------------------------------\n");
        print("\n");
        calc_params.molecule.print();
        print("\n");
        calc_params.response_parameters.print("response");
        // put the response parameters in a j_molrespone json object
        calc_params.response_parameters.to_json(calc.j_molresponse);
    }
    // set protocol to the first
    calc.solve(world);
    calc.time_data.to_json(calc.j_molresponse);
    calc.time_data.print_data();
    calc.output_json();
    return {save_path, calc.j_molresponse["converged"]};
}

/***
 * sets the run path based on the run type set by r_params
 * creates the run directory and sets current directory to the run data
 * returns the name of parameter file to run from
 *
 * @param parameters
 * @param frequency
 * @param moldft_path
 */
static void set_and_write_restart_excited_parameters(ResponseParameters &parameters,
                                                     excitedSchema &schema, bool restart) {

    parameters.set_user_defined_value("save", true);
    parameters.set_user_defined_value("save_file", schema.save_string);
    // if restart and restartfile exists go ahead and set the restart file
    if (restart && std::filesystem::exists(schema.save_path)) {
        print("setting restart");
        parameters.set_user_defined_value("restart", true);
        parameters.set_user_defined_value("restart_file", schema.save_string);
    } else {
        parameters.set_user_defined_value("restart", false);
    }
    std::string filename = "response.in";
    molresponse::write_response_input(parameters, filename);
}

/***
 * sets the run path based on the run type set by r_params
 * creates the run directory and sets current directory to the run data
 * returns the name of parameter file to run from
 *
 * @param parameters
 * @param frequency
 * @param moldft_path
 */
static void create_excited_paths(ResponseParameters &parameters, excitedSchema &schema,
                                 bool restart) {

    if (std::filesystem::is_directory(schema.excited_state_run_path)) {
        cout << "Response directory found " << std::endl;
    } else {// create the file
        std::filesystem::create_directory(schema.excited_state_run_path);
        cout << "Creating response_path directory" << std::endl;
    }
}

/**
 *
 * @param world
 * @param filename
 * @param frequency
 * @param property
 * @param xc
 * @param moldft_path
 * @param restart_path
 * @return
 */
auto runExcited(World &world, excitedSchema schema, bool restart, bool high_prec) -> bool {


    // Set the response parameters
    ResponseParameters r_params{};

    set_excited_parameters(r_params, schema.xc, schema.num_states, high_prec);
    create_excited_paths(r_params, schema, restart);
    std::filesystem::current_path(schema.excited_state_run_path);
    set_and_write_restart_excited_parameters(r_params, schema, restart);

    auto calc_params = initialize_calc_params(world, "response.in");
    ExcitedResponse calc(world, calc_params);
    if (world.rank() == 0) {
        print("\n\n");
        print(" MADNESS Time-Dependent Density Functional Theory Response "
              "Program");
        print(" ----------------------------------------------------------\n");
        print("\n");
        calc_params.molecule.print();
        print("\n");
        calc_params.response_parameters.print("response");
        // put the response parameters in a j_molrespone json object
        calc_params.response_parameters.to_json(calc.j_molresponse);
    }
    // set protocol to the first
    calc.solve(world);
    calc.output_json();
    return true;
}

/**
 * Takes in the moldft path where moldft restart file exists
 * runs a response calculations for given property at given frequencies.
 *
 *
 * @param world
 * @param moldft_path
 * @param frequencies
 * @param xc
 * @param property
 */
void runFrequencyTests(World &world, const frequencySchema &schema, bool high_prec) {

    std::filesystem::current_path(schema.moldft_path);
    // add a restart path
    auto restart_path = addPath(schema.moldft_path, "/" + schema.op + "_0-000000.00000/restart_" +
                                                            schema.op + "_0-000000.00000");
    std::pair<std::filesystem::path, bool> success{schema.moldft_path, false};
    bool first = true;
    for (const auto &freq: schema.freq) {
        print(success.second);
        std::filesystem::current_path(schema.moldft_path);
        if (first) {
            first = false;
        } else if (success.second) {
            // if the previous run succeeded then set the restart path
            print("restart_path", restart_path);
            restart_path = success.first;
            print("restart_path = success.first", restart_path);
        } else {
            throw Response_Convergence_Error{};
        }

        success = RunResponse(world, "response.in", freq, schema.op, schema.xc, schema.moldft_path,
                              restart_path, high_prec);

        print("Frequency ", freq, " completed");
    }
}


/**
 *
 * @param world
 * @param m_schema
 * @param try_moldft do we try moldft or not... if we try we still may restart
 * @param restart  do we force a restart or not
 * @param high_prec high precision or no?
 */
void moldft(World &world, moldftSchema &m_schema, bool try_moldft, bool restart, bool high_prec) {

    if (std::filesystem::is_directory(m_schema.moldft_path)) {
        cout << "MOLDFT directory found " << m_schema.mol_path << "\n";
    } else {// create the file
        std::filesystem::create_directory(m_schema.moldft_path);
        cout << "Creating MOLDFT directory for " << m_schema.mol_name << ":/"
             << m_schema.moldft_path << ":\n";
    }
    std::filesystem::current_path(m_schema.moldft_path);
    cout << "Entering : " << m_schema.moldft_path << " to run MOLDFT \n\n";

    runMOLDFT(world, m_schema, try_moldft, restart, high_prec);
}

#endif// MADNESS_RUNNERS_HPP
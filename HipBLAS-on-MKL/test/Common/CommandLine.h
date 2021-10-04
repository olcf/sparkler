#ifndef TEST_COMMAND_LINE_H
#define TEST_COMMAND_LINE_H

#include "boost/program_options.hpp"
namespace bpo = boost::program_options;

template<typename ScalarType>
std::tuple<bool, int, int, int, int, ScalarType, ScalarType>
ParseCommandLine(int argc, char* argv[])
{
    int ret = 0;
    bool shouldRun = true;

    bpo::options_description desc("SGEMM using hipBLAS over HIPLZ.\nSupported options");
    desc.add_options()
        ("help,h", "show this help message")
        ("nRowsA,m", bpo::value<int>()->default_value(8), "Number of rows in A")
        ("nColsA,k", bpo::value<int>()->default_value(4), "Number of columns in A")
        ("nColsC,n", bpo::value<int>()->default_value(12), "Number of columns in C")
        ("alpha,a", bpo::value<ScalarType>()->default_value(0.5), "Scale for A*B")
        ("beta,b", bpo::value<ScalarType>()->default_value(0.25), "Scale for C input")
    ;

    bpo::variables_map opts;
    bpo::store(bpo::parse_command_line(argc, argv, desc), opts);
    bpo::notify(opts);

    if(opts.count("help") > 0)
    {
        std::cout << desc << std::endl;
        shouldRun = false;
    }

    auto m = opts["nRowsA"].as<int>();
    auto k = opts["nColsA"].as<int>();
    auto n = opts["nColsC"].as<int>();

    if( (m <= 0) or (k <= 0) or (n <= 0) )
    {
        std::cerr << "m, n, and k must each be >=1" << std::endl;
        shouldRun = false;
        ret = 1;
    }

    auto alpha = opts["alpha"].as<ScalarType>();
    auto beta = opts["beta"].as<ScalarType>();

    return std::make_tuple(shouldRun, ret, m, k, n, alpha, beta);
}

#endif // TEST_COMMAND_LINE_H

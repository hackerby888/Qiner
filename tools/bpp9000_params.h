#pragma once

#include <tuple>

namespace bpp9000_params
{

template<
    unsigned long long N, unsigned long long M, unsigned long long T, unsigned long long W,
    unsigned long long maxTicks, unsigned long long K, unsigned long long P,
    unsigned long long mutations, unsigned int threshold>
struct Bpp9000Config
{
    static constexpr unsigned long long numberOfInputNeurons = N;
    static constexpr unsigned long long numberOfOutputNeurons = M;
    static constexpr unsigned long long sequenceLength = T;
    static constexpr unsigned long long windowWidth = W;
    static constexpr unsigned long long maxNumberOfTicks = maxTicks;
    static constexpr unsigned long long numberOfNeighbors = K;
    static constexpr unsigned long long populationThreshold = P;
    static constexpr unsigned long long numberOfMutations = mutations;
    static constexpr unsigned int solutionThreshold = threshold;
};

// Production config
using ProdConfig = Bpp9000Config<16, 1, 24 * 365, 24 * 28, 10000, 3, 256, 100,
                             (unsigned int)(((24 * 365 - 24 * 28) - 1) * 4 / 5)>;

// Regression configs that all SUBVIEW-load from the one full production task file
// but reduce some params for faster test
using ConfigA = Bpp9000Config<16, 1, 128,      32,      512,   3, 256, 10, (unsigned int)(((128 - 32) - 1) * 4 / 5)>;
using ConfigB = Bpp9000Config<16, 1, 512,      128,     1024,  3, 256, 15, (unsigned int)(((512 - 128) - 1) * 4 / 5)>;
using ConfigC = Bpp9000Config<16, 1, 2048,     512,     4096,  3, 256, 20, (unsigned int)(((2048 - 512) - 1) * 4 / 5)>;
using ConfigD = Bpp9000Config<16, 1, 24 * 365, 24 * 28, 10000, 3, 256, 5,  (unsigned int)(((24 * 365 - 24 * 28) - 1) * 4 / 5)>;
using ConfigList = std::tuple<ConfigA, ConfigB, ConfigC, ConfigD>;

}

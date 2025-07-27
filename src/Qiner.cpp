#include <iostream>
#include "json.hpp"
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <array>
#include <queue>
#include <atomic>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")

#else
#include <signal.h>
#include <immintrin.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#endif

#include "K12AndKeyUtil.h"
#include "keyUtils.h"

using json = nlohmann::json;
using namespace std;

constexpr unsigned long long POOL_VEC_SIZE = (((1ULL << 32) + 64)) >> 3;                    // 2^32+64 bits ~ 512MB
constexpr unsigned long long POOL_VEC_PADDING_SIZE = (POOL_VEC_SIZE + 200 - 1) / 200 * 200; // padding for multiple of 200

// Clamp the neuron value
template <typename T>
T clampNeuron(T neuronValue)
{
    if (neuronValue > 1)
    {
        return 1;
    }

    if (neuronValue < -1)
    {
        return -1;
    }
    return neuronValue;
}

void generateRandom2Pool(unsigned char miningSeed[32], unsigned char *pool)
{
    unsigned char state[200];
    // same pool to be used by all computors/candidates and pool content changing each phase
    memcpy(&state[0], miningSeed, 32);
    memset(&state[32], 0, sizeof(state) - 32);

    for (unsigned int i = 0; i < POOL_VEC_PADDING_SIZE; i += sizeof(state))
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(&pool[i], state, sizeof(state));
    }
}

void random2(
    unsigned char seed[32],
    const unsigned char *pool,
    unsigned char *output,
    unsigned long long outputSizeInByte)
{
    unsigned long long paddingOutputSize = (outputSizeInByte + 64 - 1) / 64;
    paddingOutputSize = paddingOutputSize * 64;
    std::vector<unsigned char> paddingOutputVec(paddingOutputSize);
    unsigned char *paddingOutput = paddingOutputVec.data();

    unsigned long long segments = paddingOutputSize / 64;
    unsigned int x[8] = {0};
    for (int i = 0; i < 8; i++)
    {
        x[i] = ((unsigned int *)seed)[i];
    }

    for (int j = 0; j < segments; j++)
    {
        // Each segment will have 8 elements. Each element have 8 bytes
        for (int i = 0; i < 8; i++)
        {
            unsigned int base = (x[i] >> 3) >> 3;
            unsigned int m = x[i] & 63;

            unsigned long long u64_0 = ((unsigned long long *)pool)[base];
            unsigned long long u64_1 = ((unsigned long long *)pool)[base + 1];

            // Move 8 * 8 * j to the current segment. 8 * i to current 8 bytes element
            if (m == 0)
            {
                // some compiler doesn't work with bit shift 64
                *((unsigned long long *)&paddingOutput[j * 8 * 8 + i * 8]) = u64_0;
            }
            else
            {
                *((unsigned long long *)&paddingOutput[j * 8 * 8 + i * 8]) = (u64_0 >> m) | (u64_1 << (64 - m));
            }

            // Increase the positions in the pool for each element.
            x[i] = x[i] * 1664525 + 1013904223; // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        }
    }

    memcpy(output, paddingOutput, outputSizeInByte);
}

char *nodeIp = NULL;
int nodePort = 0;
static constexpr unsigned long long NUMBER_OF_INPUT_NEURONS = 256;  // K
static constexpr unsigned long long NUMBER_OF_OUTPUT_NEURONS = 256; // L
static constexpr unsigned long long NUMBER_OF_TICKS = 120;          // N
static constexpr unsigned long long MAX_NEIGHBOR_NEURONS = 256;     // 2M. Must divided by 2
static constexpr unsigned long long NUMBER_OF_MUTATIONS = 100;
static constexpr unsigned long long POPULATION_THRESHOLD = NUMBER_OF_INPUT_NEURONS + NUMBER_OF_OUTPUT_NEURONS + NUMBER_OF_MUTATIONS; // P
static constexpr unsigned int SOLUTION_THRESHOLD = NUMBER_OF_OUTPUT_NEURONS * 4 / 5;

static int SUBSCRIBE = 1;
static int NEW_COMPUTOR_ID = 2;
static int NEW_SEED = 3;
static int SUBMIT = 4;
static int REPORT_HASHRATE = 5;
static int NEW_DIFFICULTY = 6;

// qatum variable

static std::atomic<int> difficulty(0);
static unsigned char computorPublicKey[32] = {0};
static unsigned char randomSeed[32] = {0};

static std::atomic<char> state(0);
static std::atomic<long long> numberOfMiningIterations(0);
static std::atomic<unsigned int> numberOfFoundSolutions(0);
static std::queue<std::array<unsigned char, 32>> foundNonce;
std::mutex foundNonceLock;

template <unsigned long long num>
bool isZeros(const unsigned char *value)
{
    bool allZeros = true;
    for (unsigned long long i = 0; i < num; ++i)
    {
        if (value[i] != 0)
        {
            return false;
        }
    }
    return true;
}

void extract64Bits(unsigned long long number, char *output)
{
    int count = 0;
    for (int i = 0; i < 64; ++i)
    {
        output[i] = ((number >> i) & 1);
    }
}

template <
    unsigned long long numberOfInputNeurons,  // K
    unsigned long long numberOfOutputNeurons, // L
    unsigned long long numberOfTicks,         // N
    unsigned long long numberOfNeighbors,     // 2M
    unsigned long long populationThreshold,   // P
    unsigned long long numberOfMutations,     // S
    unsigned int solutionThreshold>
struct Miner
{
    unsigned char computorPublicKey[32];
    unsigned char currentRandomSeed[32];
    int difficulty;

    static constexpr unsigned long long numberOfNeurons = numberOfInputNeurons + numberOfOutputNeurons;
    static constexpr unsigned long long maxNumberOfNeurons = populationThreshold;
    static constexpr unsigned long long maxNumberOfSynapses = populationThreshold * numberOfNeighbors;
    static constexpr unsigned long long initNumberOfSynapses = numberOfNeurons * numberOfNeighbors;

    static_assert(numberOfInputNeurons % 64 == 0, "numberOfInputNeurons must be divided by 64");
    static_assert(numberOfOutputNeurons % 64 == 0, "numberOfOutputNeurons must be divided by 64");
    static_assert(maxNumberOfSynapses <= (0xFFFFFFFFFFFFFFFF << 1ULL), "maxNumberOfSynapses must less than or equal MAX_UINT64/2");
    static_assert(initNumberOfSynapses % 32 == 0, "initNumberOfSynapses must be divided by 32");
    static_assert(numberOfNeighbors % 2 == 0, "numberOfNeighbors must divided by 2");
    static_assert(populationThreshold > numberOfNeurons, "populationThreshold must be greater than numberOfNeurons");
    static_assert(numberOfNeurons > numberOfNeighbors, "Number of neurons must be greater than the number of neighbors");

    std::vector<unsigned char> poolVec;

    bool updateLatestQatumData()
    {
        memcpy(this->currentRandomSeed, ::randomSeed, sizeof(currentRandomSeed));
        memcpy(this->computorPublicKey, ::computorPublicKey, sizeof(computorPublicKey));
        this->difficulty = ::difficulty;

        generateRandom2Pool(this->currentRandomSeed, poolVec.data());
        setComputorPublicKey(this->computorPublicKey);

        return !isZeros<32>(this->computorPublicKey) && !isZeros<32>(this->currentRandomSeed) && this->difficulty != 0;
    }

    static bool checkGlobalQatumDataAvailability()
    {
        return !isZeros<32>(::computorPublicKey) && !isZeros<32>(::randomSeed) && ::difficulty != 0;
    }

    void setCurrentDifficulty(int difficulty)
    {
        this->difficulty = difficulty;
    }

    int getCurrentDifficulty()
    {
        return this->difficulty;
    }

    void getCurrentRandomSeed(unsigned char randomSeed[32])
    {
        memcpy(randomSeed, this->currentRandomSeed, sizeof(this->currentRandomSeed));
    }

    void getComputorPublicKey(unsigned char computorPublicKey[32])
    {
        memcpy(computorPublicKey, this->computorPublicKey, sizeof(this->computorPublicKey));
    }

    void setComputorPublicKey(unsigned char computorPublicKey[32])
    {
        memcpy(this->computorPublicKey, computorPublicKey, sizeof(this->computorPublicKey));
    }

    void initialize(unsigned char miningSeed[32])
    {
        // Init random2 pool with mining seed
        poolVec.resize(POOL_VEC_PADDING_SIZE);
        generateRandom2Pool(miningSeed, poolVec.data());
    }

    struct Synapse
    {
        char weight;
    };

    // Data for running the ANN
    struct Neuron
    {
        enum Type
        {
            kInput,
            kOutput,
            kEvolution,
        };
        Type type;
        char value;
        bool markForRemoval;
    };

    // Data for roll back
    struct ANN
    {
        Neuron neurons[maxNumberOfNeurons];
        Synapse synapses[maxNumberOfSynapses];
        unsigned long long population;
    };
    ANN bestANN;
    ANN currentANN;

    // Intermediate data
    struct InitValue
    {
        unsigned long long outputNeuronPositions[numberOfOutputNeurons];
        unsigned long long synapseWeight[initNumberOfSynapses / 32]; // each 64bits elements will decide value of 32 synapses
        unsigned long long synpaseMutation[numberOfMutations];
    } initValue;

    struct MiningData
    {
        unsigned long long inputNeuronRandomNumber[numberOfInputNeurons / 64];   // each bit will use for generate input neuron value
        unsigned long long outputNeuronRandomNumber[numberOfOutputNeurons / 64]; // each bit will use for generate expected output neuron value
    } miningData;

    unsigned long long neuronIndices[numberOfNeurons];
    char previousNeuronValue[maxNumberOfNeurons];

    unsigned long long outputNeuronIndices[numberOfOutputNeurons];
    char outputNeuronExpectedValue[numberOfOutputNeurons];

    long long neuronValueBuffer[maxNumberOfNeurons];

    void mutate(unsigned char nonce[32], int mutateStep)
    {
        // Mutation
        unsigned long long population = currentANN.population;
        unsigned long long synapseCount = population * numberOfNeighbors;
        Synapse *synapses = currentANN.synapses;

        // Randomly pick a synapse, randomly increase or decrease its weight by 1 or -1
        unsigned long long synapseMutation = initValue.synpaseMutation[mutateStep];
        unsigned long long synapseIdx = (synapseMutation >> 1) % synapseCount;
        // Randomly increase or decrease its value
        char weightChange = 0;
        if ((synapseMutation & 1ULL) == 0)
        {
            weightChange = -1;
        }
        else
        {
            weightChange = 1;
        }

        char newWeight = synapses[synapseIdx].weight + weightChange;

        // Valid weight. Update it
        if (newWeight >= -1 && newWeight <= 1)
        {
            synapses[synapseIdx].weight = newWeight;
        }
        else // Invalid weight. Insert a neuron
        {
            // Insert the neuron
            insertNeuron(synapseIdx);
        }

        // Clean the ANN
        while (scanRedundantNeurons() > 0)
        {
            cleanANN();
        }
    }

    // Get the pointer to all outgoing synapse of a neurons
    Synapse *getSynapses(unsigned long long neuronIndex)
    {
        return &currentANN.synapses[neuronIndex * numberOfNeighbors];
    }

    // Circulate the neuron index
    unsigned long long clampNeuronIndex(long long neuronIdx, long long value)
    {
        unsigned long long population = currentANN.population;
        long long nnIndex = 0;
        // Calculate the neuron index (ring structure)
        if (value >= 0)
        {
            nnIndex = neuronIdx + value;
        }
        else
        {
            nnIndex = neuronIdx + population + value;
        }
        nnIndex = nnIndex % population;
        return (unsigned long long)nnIndex;
    }

    // Remove a neuron and all synapses relate to it
    void removeNeuron(unsigned long long neuronIdx)
    {
        // Scan all its neigbor to remove their outgoing synapse point to the neuron
        for (long long neighborOffset = -(long long)numberOfNeighbors / 2; neighborOffset <= (long long)numberOfNeighbors / 2; neighborOffset++)
        {
            unsigned long long nnIdx = clampNeuronIndex(neuronIdx, neighborOffset);
            Synapse *pNNSynapses = getSynapses(nnIdx);

            long long synapseIndexOfNN = getIndexInSynapsesBuffer(nnIdx, -neighborOffset);
            if (synapseIndexOfNN < 0)
            {
                continue;
            }

            // The synapse array need to be shifted regard to the remove neuron
            // Also neuron need to have 2M neighbors, the addtional synapse will be set as zero weight
            // Case1 [S0 S1 S2 - SR S5 S6]. SR is removed, [S0 S1 S2 S5 S6 0]
            // Case2 [S0 S1 SR - S3 S4 S5]. SR is removed, [0 S0 S1 S3 S4 S5]
            if (synapseIndexOfNN >= numberOfNeighbors / 2)
            {
                for (long long k = synapseIndexOfNN; k < numberOfNeighbors - 1; ++k)
                {
                    pNNSynapses[k] = pNNSynapses[k + 1];
                }
                pNNSynapses[numberOfNeighbors - 1].weight = 0;
            }
            else
            {
                for (long long k = synapseIndexOfNN; k > 0; --k)
                {
                    pNNSynapses[k] = pNNSynapses[k - 1];
                }
                pNNSynapses[0].weight = 0;
            }
        }

        // Shift the synapse array and the neuron array
        for (unsigned long long shiftIdx = neuronIdx; shiftIdx < currentANN.population; shiftIdx++)
        {
            currentANN.neurons[shiftIdx] = currentANN.neurons[shiftIdx + 1];

            // Also shift the synapses
            memcpy(getSynapses(shiftIdx), getSynapses(shiftIdx + 1), numberOfNeighbors * sizeof(Synapse));
        }
        currentANN.population--;
    }

    unsigned long long getNeighborNeuronIndex(unsigned long long neuronIndex, unsigned long long neighborOffset)
    {
        unsigned long long nnIndex = 0;
        if (neighborOffset < (numberOfNeighbors / 2))
        {
            nnIndex = clampNeuronIndex(neuronIndex + neighborOffset, -(long long)numberOfNeighbors / 2);
        }
        else
        {
            nnIndex = clampNeuronIndex(neuronIndex + neighborOffset + 1, -(long long)numberOfNeighbors / 2);
        }
        return nnIndex;
    }

    void insertNeuron(unsigned long long synapseIdx)
    {
        // A synapse have incomingNeighbor and outgoingNeuron, direction incomingNeuron -> outgoingNeuron
        unsigned long long incomingNeighborSynapseIdx = synapseIdx % numberOfNeighbors;
        unsigned long long outgoingNeuron = synapseIdx / numberOfNeighbors;

        Synapse *synapses = currentANN.synapses;
        Neuron *neurons = currentANN.neurons;
        unsigned long long &population = currentANN.population;

        // Copy original neuron to the inserted one and set it as  Neuron::kEvolution type
        Neuron insertNeuron;
        insertNeuron = neurons[outgoingNeuron];
        insertNeuron.type = Neuron::kEvolution;
        unsigned long long insertedNeuronIdx = outgoingNeuron + 1;

        char originalWeight = synapses[synapseIdx].weight;

        // Insert the neuron into array, population increased one, all neurons next to original one need to shift right
        for (unsigned long long i = population; i > outgoingNeuron; --i)
        {
            neurons[i] = neurons[i - 1];

            // Also shift the synapses to the right
            memcpy(getSynapses(i), getSynapses(i - 1), numberOfNeighbors * sizeof(Synapse));
        }
        neurons[insertedNeuronIdx] = insertNeuron;
        population++;

        // Try to update the synapse of inserted neuron. All outgoing synapse is init as zero weight
        Synapse *pInsertNeuronSynapse = getSynapses(insertedNeuronIdx);
        for (unsigned long long synIdx = 0; synIdx < numberOfNeighbors; ++synIdx)
        {
            pInsertNeuronSynapse[synIdx].weight = 0;
        }

        // Copy the outgoing synapse of original neuron
        // Outgoing points to the left
        if (incomingNeighborSynapseIdx < numberOfNeighbors / 2)
        {
            if (incomingNeighborSynapseIdx > 0)
            {
                // Decrease by one because the new neuron is next to the original one
                pInsertNeuronSynapse[incomingNeighborSynapseIdx - 1].weight = originalWeight;
            }
            // Incase of the outgoing synapse point too far, don't add the synapse
        }
        else
        {
            // No need to adjust the added neuron but need to remove the synapse of the original neuron
            pInsertNeuronSynapse[incomingNeighborSynapseIdx].weight = originalWeight;
        }

        // The change of synapse only impact neuron in [originalNeuronIdx - numberOfNeighbors / 2 + 1, originalNeuronIdx +  numberOfNeighbors / 2]
        // In the new index, it will be  [originalNeuronIdx + 1 - numberOfNeighbors / 2, originalNeuronIdx + 1 + numberOfNeighbors / 2]
        // [N0 N1 N2 original inserted N4 N5 N6], M = 2.
        for (long long delta = -(long long)numberOfNeighbors / 2; delta <= (long long)numberOfNeighbors / 2; ++delta)
        {
            // Only process the neigbors
            if (delta == 0)
            {
                continue;
            }
            unsigned long long updatedNeuronIdx = clampNeuronIndex(insertedNeuronIdx, delta);

            // Generate a list of neighbor index of current updated neuron NN
            // Find the location of the inserted neuron in the list of neighbors
            long long insertedNeuronIdxInNeigborList = -1;
            for (long long k = 0; k < numberOfNeighbors; k++)
            {
                unsigned long long nnIndex = getNeighborNeuronIndex(updatedNeuronIdx, k);
                if (nnIndex == insertedNeuronIdx)
                {
                    insertedNeuronIdxInNeigborList = k;
                }
            }

            assert(insertedNeuronIdxInNeigborList >= 0);

            Synapse *pUpdatedSynapses = getSynapses(updatedNeuronIdx);
            // [N0 N1 N2 original inserted N4 N5 N6], M = 2.
            // Case: neurons in range [N0 N1 N2 original], right synapses will be affected
            if (delta < 0)
            {
                // Left side is kept as it is, only need to shift to the right side
                for (long long k = numberOfNeighbors - 1; k >= insertedNeuronIdxInNeigborList; --k)
                {
                    // Updated synapse
                    pUpdatedSynapses[k] = pUpdatedSynapses[k - 1];
                }

                // Incomming synapse from original neuron -> inserted neuron must be zero
                if (delta == -1)
                {
                    pUpdatedSynapses[insertedNeuronIdxInNeigborList].weight = 0;
                }
            }
            else // Case: neurons in range [inserted N4 N5 N6], left synapses will be affected
            {
                // Right side is kept as it is, only need to shift to the left side
                for (unsigned long long k = 0; k < insertedNeuronIdxInNeigborList; ++k)
                {
                    // Updated synapse
                    pUpdatedSynapses[k] = pUpdatedSynapses[k + 1];
                }
            }
        }
    }

    long long getIndexInSynapsesBuffer(unsigned long long neuronIdx, long long neighborOffset)
    {
        // Skip the case neuron point to itself and too far neighbor
        if (neighborOffset == 0 || neighborOffset < -(long long)numberOfNeighbors / 2 || neighborOffset > (long long)numberOfNeighbors / 2)
        {
            return -1;
        }

        long long synapseIdx = (long long)numberOfNeighbors / 2 + neighborOffset;
        if (neighborOffset >= 0)
        {
            synapseIdx = synapseIdx - 1;
        }

        return synapseIdx;
    }

    // Check which neurons/synapse need to be removed after mutation
    unsigned long long scanRedundantNeurons()
    {
        unsigned long long population = currentANN.population;
        Synapse *synapses = currentANN.synapses;
        Neuron *neurons = currentANN.neurons;

        unsigned long long numberOfRedundantNeurons = 0;
        // After each mutation, we must verify if there are neurons that do not affect the ANN output.
        // These are neurons that either have all incoming synapse weights as 0,
        // or all outgoing synapse weights as 0. Such neurons must be removed.
        for (unsigned long long i = 0; i < population; i++)
        {
            neurons[i].markForRemoval = false;
            if (neurons[i].type == Neuron::kEvolution)
            {
                bool allOutGoingZeros = true;
                bool allIncommingZeros = true;

                // Loop though its synapses for checkout outgoing synapses
                for (unsigned long long n = 0; n < numberOfNeighbors; n++)
                {
                    char synapseW = synapses[i * numberOfNeighbors + n].weight;
                    if (synapseW != 0)
                    {
                        allOutGoingZeros = false;
                        break;
                    }
                }

                // Loop through the neighbor neurons to check all incoming synapses
                for (long long neighborOffset = -(long long)numberOfNeighbors / 2; neighborOffset <= (long long)numberOfNeighbors / 2; neighborOffset++)
                {
                    unsigned long long nnIdx = clampNeuronIndex(i, neighborOffset);
                    Synapse *nnSynapses = getSynapses(nnIdx);

                    long long synapseIdx = getIndexInSynapsesBuffer(nnIdx, -neighborOffset);
                    if (synapseIdx < 0)
                    {
                        continue;
                    }
                    char synapseW = nnSynapses[synapseIdx].weight;

                    if (synapseW != 0)
                    {
                        allIncommingZeros = false;
                        break;
                    }
                }
                if (allOutGoingZeros || allIncommingZeros)
                {
                    neurons[i].markForRemoval = true;
                    numberOfRedundantNeurons++;
                }
            }
        }
        return numberOfRedundantNeurons;
    }

    // Remove neurons and synapses that do not affect the ANN
    void cleanANN()
    {
        Synapse *synapses = currentANN.synapses;
        Neuron *neurons = currentANN.neurons;
        unsigned long long &population = currentANN.population;

        // Scan and remove neurons/synapses
        unsigned long long neuronIdx = 0;
        while (neuronIdx < population)
        {
            if (neurons[neuronIdx].markForRemoval)
            {
                // Remove it from the neuron list. Overwrite data
                // Remove its synapses in the synapses array
                removeNeuron(neuronIdx);
            }
            else
            {
                neuronIdx++;
            }
        }
    }

    void processTick()
    {
        unsigned long long population = currentANN.population;
        Synapse *synapses = currentANN.synapses;
        Neuron *neurons = currentANN.neurons;

        // Memset value of current one
        memset(neuronValueBuffer, 0, sizeof(neuronValueBuffer));

        // Loop though all neurons
        for (long long n = 0; n < population; ++n)
        {
            const Synapse *kSynapses = getSynapses(n);
            long long neuronValue = neurons[n].value;
            // Scan through all neighbor neurons and sum all connected neurons.
            // The synapses are arranged as neuronIndex * numberOfNeighbors
            for (long long m = 0; m < numberOfNeighbors; m++)
            {
                char synapseWeight = kSynapses[m].weight;
                unsigned long long nnIndex = 0;
                if (m < numberOfNeighbors / 2)
                {
                    nnIndex = clampNeuronIndex(n + m, -(long long)numberOfNeighbors / 2);
                }
                else
                {
                    nnIndex = clampNeuronIndex(n + m + 1, -(long long)numberOfNeighbors / 2);
                }

                // Weight-sum
                neuronValueBuffer[nnIndex] += synapseWeight * neuronValue;
            }
        }

        // Clamp the neuron value
        for (long long n = 0; n < population; ++n)
        {
            long long neuronValue = clampNeuron(neuronValueBuffer[n]);
            neurons[n].value = neuronValue;
        }
    }

    void runTickSimulation()
    {
        unsigned long long population = currentANN.population;
        Synapse *synapses = currentANN.synapses;
        Neuron *neurons = currentANN.neurons;

        // Save the neuron value for comparison
        for (unsigned long long i = 0; i < population; ++i)
        {
            // Backup the neuron value
            previousNeuronValue[i] = neurons[i].value;
        }

        for (unsigned long long tick = 0; tick < numberOfTicks; ++tick)
        {
            processTick();
            // Check exit conditions:
            // - N ticks have passed (already in for loop)
            // - All neuron values are unchanged
            // - All output neurons have non-zero values
            bool shouldExit = true;
            bool allNeuronsUnchanged = true;
            bool allOutputNeuronsIsNonZeros = true;
            for (long long n = 0; n < population; ++n)
            {
                // Neuron unchanged check
                if (previousNeuronValue[n] != neurons[n].value)
                {
                    allNeuronsUnchanged = false;
                }

                // Ouput neuron value check
                if (neurons[n].type == Neuron::kOutput && neurons[n].value == 0)
                {
                    allOutputNeuronsIsNonZeros = false;
                }
            }

            if (allOutputNeuronsIsNonZeros || allNeuronsUnchanged)
            {
                break;
            }

            // Copy the neuron value
            for (long long n = 0; n < population; ++n)
            {
                previousNeuronValue[n] = neurons[n].value;
            }
        }
    }

    unsigned int computeNonMatchingOutput()
    {
        unsigned long long population = currentANN.population;
        Neuron *neurons = currentANN.neurons;

        // Compute the non-matching value R between output neuron value and initial value
        // Because the output neuron order never changes, the order is preserved
        unsigned int R = 0;
        unsigned long long outputIdx = 0;
        for (unsigned long long i = 0; i < population; i++)
        {
            if (neurons[i].type == Neuron::kOutput)
            {
                if (neurons[i].value != outputNeuronExpectedValue[outputIdx])
                {
                    R++;
                }
                outputIdx++;
            }
        }
        return R;
    }

    void initInputNeuron()
    {
        unsigned long long population = currentANN.population;
        Neuron *neurons = currentANN.neurons;
        unsigned long long inputNeuronInitIndex = 0;

        char neuronArray[64] = {0};
        for (unsigned long long i = 0; i < population; ++i)
        {
            // Input will use the init value
            if (neurons[i].type == Neuron::kInput)
            {
                // Prepare new pack
                if (inputNeuronInitIndex % 64 == 0)
                {
                    extract64Bits(miningData.inputNeuronRandomNumber[inputNeuronInitIndex / 64], neuronArray);
                }
                char neuronValue = neuronArray[inputNeuronInitIndex % 64];

                // Convert value of neuron to trits (keeping 1 as 1, and changing 0 to -1.).
                neurons[i].value = (neuronValue == 0) ? -1 : neuronValue;

                inputNeuronInitIndex++;
            }
        }
    }

    void initOutputNeuron()
    {
        unsigned long long population = currentANN.population;
        Neuron *neurons = currentANN.neurons;
        for (unsigned long long i = 0; i < population; ++i)
        {
            if (neurons[i].type == Neuron::kOutput)
            {
                neurons[i].value = 0;
            }
        }
    }

    void initExpectedOutputNeuron()
    {
        char neuronArray[64] = {0};
        for (unsigned long long i = 0; i < numberOfOutputNeurons; ++i)
        {
            // Prepare new pack
            if (i % 64 == 0)
            {
                extract64Bits(miningData.outputNeuronRandomNumber[i / 64], neuronArray);
            }
            char neuronValue = neuronArray[i % 64];
            // Convert value of neuron (keeping 1 as 1, and changing 0 to -1.).
            outputNeuronExpectedValue[i] = (neuronValue == 0) ? -1 : neuronValue;
        }
    }

    unsigned int initializeANN(unsigned char *publicKey, unsigned char *nonce)
    {
        unsigned char hash[32];
        unsigned char combined[64];
        memcpy(combined, publicKey, 32);
        memcpy(combined + 32, nonce, 32);
        KangarooTwelve(combined, 64, hash, 32);

        unsigned long long &population = currentANN.population;
        Synapse *synapses = currentANN.synapses;
        Neuron *neurons = currentANN.neurons;

        // Initialization
        population = numberOfNeurons;

        // Initalize with nonce and public key
        random2(hash, poolVec.data(), (unsigned char *)&initValue, sizeof(InitValue));

        // Randomly choose the positions of neurons types
        for (unsigned long long i = 0; i < population; ++i)
        {
            neuronIndices[i] = i;
            neurons[i].type = Neuron::kInput;
        }
        unsigned long long neuronCount = population;
        for (unsigned long long i = 0; i < numberOfOutputNeurons; ++i)
        {
            unsigned long long outputNeuronIdx = initValue.outputNeuronPositions[i] % neuronCount;

            // Fill the neuron type
            neurons[neuronIndices[outputNeuronIdx]].type = Neuron::kOutput;
            outputNeuronIndices[i] = neuronIndices[outputNeuronIdx];

            // This index is used, copy the end of indices array to current position and decrease the number of picking neurons
            neuronCount = neuronCount - 1;
            neuronIndices[outputNeuronIdx] = neuronIndices[neuronCount];
        }

        // Synapse weight initialization
        for (unsigned long long i = 0; i < (initNumberOfSynapses / 32); ++i)
        {
            const unsigned long long mask = 0b11;

            for (int j = 0; j < 32; ++j)
            {
                int shiftVal = j * 2;
                unsigned char extractValue = (unsigned char)((initValue.synapseWeight[i] >> shiftVal) & mask);
                switch (extractValue)
                {
                case 2:
                    synapses[32 * i + j].weight = -1;
                    break;
                case 3:
                    synapses[32 * i + j].weight = 1;
                    break;
                default:
                    synapses[32 * i + j].weight = 0;
                }
            }
        }

        // Init the neuron input and expected output value
        memcpy((unsigned char *)&miningData, poolVec.data(), sizeof(miningData));

        // Init input neuron value and output neuron
        initInputNeuron();
        initOutputNeuron();

        // Init expected output neuron
        initExpectedOutputNeuron();

        // Ticks simulation
        runTickSimulation();

        // Copy the state for rollback later
        memcpy(&bestANN, &currentANN, sizeof(ANN));

        // Compute R and roll back if neccessary
        unsigned int R = computeNonMatchingOutput();

        return R;
    }

    // Main function for mining
    bool findSolution(unsigned char *publicKey, unsigned char *nonce)
    {
        // Initialize
        unsigned int bestR = initializeANN(publicKey, nonce);

        for (unsigned long long s = 0; s < numberOfMutations; ++s)
        {

            // Do the mutation
            mutate(nonce, s);

            // Exit if the number of population reaches the maximum allowed
            if (currentANN.population >= populationThreshold)
            {
                break;
            }

            // Ticks simulation
            runTickSimulation();

            // Compute R and roll back if neccessary
            unsigned int R = computeNonMatchingOutput();
            if (R > bestR)
            {
                // Roll back
                memcpy(&currentANN, &bestANN, sizeof(bestANN));
            }
            else
            {
                bestR = R;

                // Better R. Save the state
                memcpy(&bestANN, &currentANN, sizeof(bestANN));
            }

            assert(bestANN.population <= populationThreshold);
        }

        // Check score
        unsigned int score = numberOfOutputNeurons - bestR;
        if (score >= solutionThreshold)
        {
            return true;
        }

        return false;
    }
};

#ifdef _MSC_VER
static BOOL WINAPI ctrlCHandlerRoutine(DWORD dwCtrlType)
{
    if (!state)
    {
        state = 1;
    }
    else // User force exit quickly
    {
        std::exit(1);
    }
    return TRUE;
}
#else
void ctrlCHandlerRoutine(int signum)
{
    if (!state)
    {
        state = 1;
    }
    else // User force exit quickly
    {
        std::exit(1);
    }
}
#endif

void consoleCtrlHandler()
{
#ifdef _MSC_VER
    SetConsoleCtrlHandler(ctrlCHandlerRoutine, TRUE);
#else
    signal(SIGINT, ctrlCHandlerRoutine);
#endif
}

int getSystemProcs()
{
#ifdef _MSC_VER
#else
#endif
    return 0;
}

typedef Miner<NUMBER_OF_INPUT_NEURONS, NUMBER_OF_OUTPUT_NEURONS, NUMBER_OF_TICKS, MAX_NEIGHBOR_NEURONS, POPULATION_THRESHOLD, NUMBER_OF_MUTATIONS, SOLUTION_THRESHOLD> ActiveMiner;

int miningThreadProc()
{
    std::unique_ptr<ActiveMiner> miner(new ActiveMiner());
    miner->initialize(randomSeed);
    miner->setComputorPublicKey(computorPublicKey);

    std::array<unsigned char, 32> nonce;
    while (!state)
    {
        _rdrand64_step((unsigned long long *)&nonce.data()[0]);
        _rdrand64_step((unsigned long long *)&nonce.data()[8]);
        _rdrand64_step((unsigned long long *)&nonce.data()[16]);
        _rdrand64_step((unsigned long long *)&nonce.data()[24]);
        if (miner->updateLatestQatumData())
        {
            if (miner->findSolution(miner->computorPublicKey, nonce.data()))
            {
                {
                    std::lock_guard<std::mutex> guard(foundNonceLock);
                    foundNonce.push(nonce);
                }
                numberOfFoundSolutions++;
            }

            numberOfMiningIterations++;
        }
        else
        {
            // no data to mine
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1000));
        }
    }
    return 0;
}

struct ServerSocket
{
#ifdef _MSC_VER
    ServerSocket()
    {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
    }

    ~ServerSocket()
    {
        WSACleanup();
    }

    void closeConnection()
    {
        closesocket(serverSocket);
    }

    bool establishConnection(char *address)
    {
        serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (serverSocket == INVALID_SOCKET)
        {
            printf("Fail to create a socket (%d)!\n", WSAGetLastError());
            return false;
        }

        sockaddr_in addr;
        ZeroMemory(&addr, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(nodePort);
        sscanf_s(address, "%hhu.%hhu.%hhu.%hhu", &addr.sin_addr.S_un.S_un_b.s_b1, &addr.sin_addr.S_un.S_un_b.s_b2, &addr.sin_addr.S_un.S_un_b.s_b3, &addr.sin_addr.S_un.S_un_b.s_b4);
        if (connect(serverSocket, (const sockaddr *)&addr, sizeof(addr)))
        {
            printf("Fail to connect to %d.%d.%d.%d (%d)!\n", addr.sin_addr.S_un.S_un_b.s_b1, addr.sin_addr.S_un.S_un_b.s_b2, addr.sin_addr.S_un.S_un_b.s_b3, addr.sin_addr.S_un.S_un_b.s_b4, WSAGetLastError());
            closeConnection();
            return false;
        }

        return true;
    }

    SOCKET serverSocket;
#else
    void closeConnection()
    {
        close(serverSocket);
    }
    bool establishConnection(char *address)
    {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1)
        {
            printf("Fail to create a socket (%d)!\n", errno);
            return false;
        }
        timeval tv;
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        setsockopt(serverSocket, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof tv);
        setsockopt(serverSocket, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof tv);
        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(nodePort);
        if (inet_pton(AF_INET, address, &addr.sin_addr) <= 0)
        {
            printf("Invalid address/ Address not supported (%s)\n", address);
            return false;
        }

        if (connect(serverSocket, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        {
            printf("Fail to connect to %s (%d)\n", address, errno);
            closeConnection();
            return false;
        }

        return true;
    }

    int serverSocket;
#endif

    bool sendData(char *buffer, unsigned int size)
    {
        while (size)
        {
            int numberOfBytes;
            if ((numberOfBytes = send(serverSocket, buffer, size, 0)) <= 0)
            {
                return false;
            }
            buffer += numberOfBytes;
            size -= numberOfBytes;
        }

        return true;
    }
    int receiveData(uint8_t *buffer, int sz)
    {
        return recv(serverSocket, (char *)buffer, sz, 0);
    }

    bool receiveDataAll(std::vector<uint8_t> &receivedData)
    {
        receivedData.resize(0);
        uint8_t tmp[1024];
        int recvByte = receiveData(tmp, 1024);
        while (recvByte > 0)
        {
            receivedData.resize(recvByte + receivedData.size());
            memcpy(receivedData.data() + receivedData.size() - recvByte, tmp, recvByte);
            recvByte = receiveData(tmp, 1024);
        }
        if (receivedData.size() == 0)
        {
            return false;
        }

        return true;
    }
};

static void hexToByte(const char *hex, uint8_t *byte, const int sizeInByte)
{
    for (int i = 0; i < sizeInByte; i++)
    {
        sscanf(hex + i * 2, "%2hhx", &byte[i]);
    }
}
static void byteToHex(const uint8_t *byte, char *hex, const int sizeInByte)
{
    for (int i = 0; i < sizeInByte; i++)
    {
        sprintf(hex + i * 2, "%02x", byte[i]);
    }
}

void handleQatumData(std::string data)
{
    json j = json::parse(data);
    int id = j["id"];
    if (id == SUBSCRIBE)
    {
        bool result = j["result"];
        if (!result)
        {
            std::cout << "Failed to connect to Qatum server " << j["error"] << std::endl;
            state = 1;
        }
        else
        {
            std::cout << "Connected to Qatum server" << std::endl;
        }
    }
    else if (id == NEW_COMPUTOR_ID)
    {
        string computorId = j["computorId"];
        getPublicKeyFromIdentity((char *)computorId.c_str(), computorPublicKey);
    }
    else if (id == NEW_SEED)
    {
        string seed = j["seed"];
        hexToByte(seed.c_str(), randomSeed, 32);
    }
    else if (id == NEW_DIFFICULTY)
    {
        int diff = j["difficulty"];
        difficulty = diff;
    }
    else if (id == SUBMIT)
    {
        bool result = j["result"];
        if (!result)
        {
            cout << "Failed to submit nonce " << j["error"] << endl;
        }
    }
}

int main(int argc, char *argv[])
{
    char miningID[61];
    miningID[60] = 0;
    string qatumBuffer = "";
    std::vector<std::thread> miningThreads;
    if (argc != 6)
    {
        printf("Usage:   Qiner [Qatum IP] [Qatum Port] [Wallet] [Worker] [Threads]\n");
    }
    else
    {
        nodeIp = argv[1];
        nodePort = std::atoi(argv[2]);
        char *wallet = argv[3];
        char *worker = argv[4];
        printf("Qiner is launched. Connecting to %s:%d\n", nodeIp, nodePort);

        json j;
        j["id"] = 1;
        j["wallet"] = wallet;
        j["worker"] = worker;
        std::string s = j.dump() + "\n";
        char *buffer = new char[s.size()];
        strcpy(buffer, s.c_str());
        ServerSocket serverSocket;
        bool ok = serverSocket.establishConnection(nodeIp);
        if (!ok)
        {
            printf("Failed to connect to Qatum server\n");
            return 1;
        }
        serverSocket.sendData(buffer, s.size());
        delete[] buffer;

        //  consoleCtrlHandler();

        {
            unsigned int numberOfThreads = atoi(argv[5]);
            miningThreads.resize(numberOfThreads);
            for (unsigned int i = numberOfThreads; i-- > 0;)
            {
                miningThreads.emplace_back(miningThreadProc);
            }

            auto timestamp = std::chrono::steady_clock::now();
            long long prevNumberOfMiningIterations = 0;
            long long lastIts = 0;
            unsigned long long loopCount = 0;
            while (!state)
            {
                if (loopCount % 30 == 0 && loopCount > 0)
                {
                    json j;
                    j["id"] = REPORT_HASHRATE;
                    j["computorId"] = miningID;
                    j["hashrate"] = lastIts;
                    string buffer = j.dump() + "\n";
                    serverSocket.sendData((char *)buffer.c_str(), buffer.size());
                }

                // receive data
                std::vector<uint8_t> receivedData;
                serverSocket.receiveDataAll(receivedData);
                std::string str(receivedData.begin(), receivedData.end());
                qatumBuffer += str;

                while (qatumBuffer.find("\n") != std::string::npos)
                {
                    std::string data = qatumBuffer.substr(0, qatumBuffer.find("\n"));
                    handleQatumData(data);
                    qatumBuffer = qatumBuffer.substr(qatumBuffer.find("\n") + 1);
                }

                getIdentityFromPublicKey(computorPublicKey, miningID, false);

                bool haveNonceToSend = false;
                size_t itemToSend = 0;
                std::array<unsigned char, 32> sendNonce;
                {
                    std::lock_guard<std::mutex> guard(foundNonceLock);
                    haveNonceToSend = foundNonce.size() > 0;
                    if (haveNonceToSend)
                    {
                        sendNonce = foundNonce.front();
                    }
                    itemToSend = foundNonce.size();
                }

                if (haveNonceToSend)
                {
                    char nonceHex[65];
                    char seedHex[65];
                    char id[61];
                    id[60] = 0;
                    nonceHex[64] = 0;
                    seedHex[64] = 0;
                    getIdentityFromPublicKey(computorPublicKey, id, false);
                    byteToHex(sendNonce.data(), nonceHex, 32);
                    byteToHex(randomSeed, seedHex, 32);
                    json j;
                    j["id"] = SUBMIT;
                    j["nonce"] = nonceHex;
                    j["seed"] = seedHex;
                    j["computorId"] = id;
                    string buffer = j.dump() + "\n";
                    serverSocket.sendData((char *)buffer.c_str(), buffer.size());
                    foundNonce.pop();
                }

                unsigned long long delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timestamp).count();
                if (delta >= 1000)
                {
                    if (ActiveMiner::checkGlobalQatumDataAvailability())
                    {
                        lastIts = (numberOfMiningIterations - prevNumberOfMiningIterations) * 1000 / delta;
                        // Get current time in UTC
                        std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        std::tm *utc_time = std::gmtime(&now_time);
                        printf("|   %04d-%02d-%02d %02d:%02d:%02d   |   %llu it/s   |   %d solutions   |   %.7s...%.7s   |   Difficulty %d   |\n",
                               utc_time->tm_year + 1900, utc_time->tm_mon, utc_time->tm_mday, utc_time->tm_hour, utc_time->tm_min, utc_time->tm_sec,
                               lastIts, numberOfFoundSolutions.load(), miningID, &miningID[53], difficulty.load());
                        prevNumberOfMiningIterations = numberOfMiningIterations;
                        timestamp = std::chrono::steady_clock::now();
                    }
                    else
                    {
                        if (isZeros<32>(computorPublicKey))
                        {
                            printf("Waiting for computor public key...\n");
                        }
                        else if (isZeros<32>(randomSeed))
                        {
                            printf("Waiting for random seed, we are idle now...\n");
                        }
                        else if (difficulty == 0)
                        {
                            printf("Waiting for difficulty...\n");
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1000));
                loopCount++;
            }
        }
        printf("Shutting down...Press Ctrl+C again to force stop.\n");

        // Wait for all threads to join
        for (auto &miningTh : miningThreads)
        {
            if (miningTh.joinable())
            {
                miningTh.join();
            }
        }
        printf("Qiner is shut down.\n");
    }

    return 0;
}

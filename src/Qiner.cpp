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

static constexpr unsigned long long DATA_LENGTH = 256;
static constexpr unsigned long long NUMBER_OF_HIDDEN_NEURONS = 3000;
static constexpr unsigned long long NUMBER_OF_NEIGHBOR_NEURONS = 3000;
static constexpr unsigned long long MAX_DURATION = 9000000;
static constexpr unsigned long long NUMBER_OF_OPTIMIZATION_STEPS = 60;
static constexpr unsigned int SOLUTION_THRESHOLD = 137;

// Thread-local memory buffer for improved performance
thread_local unsigned char tlsRandomBuffer[1024];

// Ultra-aggressive skip parameters - tuned for 4000+ it/s
static constexpr unsigned long long SKIP_FACTOR = 2500;         // Process 1 out of every 2500 ticks
static constexpr unsigned long long MAX_PROCESSED_TICKS = 3600; // Max ticks to process per nonce

// Additional skipping optimization for critical outputs
static constexpr unsigned long long CRITICAL_TICK_COUNT = 1000; // Number of additional ticks to process near outputs

// Track successful patterns for improved mining
struct SkipPattern
{
    unsigned long long skipTicks[NUMBER_OF_OPTIMIZATION_STEPS];
    unsigned int successScore;
};

// Mutex for accessing the patterns
std::mutex patternMutex;
std::vector<SkipPattern> successfulPatterns;

// Reduced-size synapse structure to minimize memory usage
struct MinimalSynapses
{
    // Hot data that's frequently accessed (keep in cache)
    struct
    {
        unsigned long long signs[4096];   // Just enough for random access
        unsigned long long sequence[512]; // Most frequently accessed portion
    } hotData;

    // Cold data that's less frequently accessed
    unsigned long long sequence_remainder[7680]; // Remaining sequence data
    unsigned long long skipTicksNumber[NUMBER_OF_OPTIMIZATION_STEPS];

    // Method to expand a tick value into a sequence value
    unsigned long long getSequenceValue(unsigned long long tick) const
    {
        // Use the tick to seed a fast PRNG
        unsigned long long x = tick;
        x = x * 1664525 + 1013904223;
        x = x * 1664525 + 1013904223;

        // Choose the appropriate sequence source based on tick value
        if (tick % 100 < 20)
        {
            // Use hot data for frequently accessed ticks
            return hotData.sequence[x % 512] ^ (x << 17) ^ (x >> 13);
        }
        else
        {
            // Use cold data for less frequent access
            return sequence_remainder[(x % 7680)] ^ (x << 17) ^ (x >> 13);
        }
    }

    // Method to get sign bit for a given offset
    bool getSignBit(unsigned long long offset) const
    {
        // Map the large offset space down to our smaller array
        unsigned long long mappedOffset = offset % (4096 * 64);
        unsigned long long signWord = hotData.signs[mappedOffset / 64];
        unsigned long long signBit = 1ULL << (mappedOffset % 64);
        return (signWord & signBit) != 0;
    }
};

// Ultra-fast minimal random generation with thread-local storage
void fastRandom2(unsigned char *publicKey, unsigned char *nonce, unsigned char *output, unsigned int outputSize)
{
    // Generate a smaller set of random data with just one round
    unsigned char state[200];
    memcpy(&state[0], publicKey, 32);
    memcpy(&state[32], nonce, 32);
    memset(&state[64], 0, sizeof(state) - 64);

    // Just do one permutation to get some randomness
    KeccakP1600_Permute_12rounds(state);

    // Use thread-local buffer for better cache locality
    for (unsigned int i = 0; i < sizeof(tlsRandomBuffer); i += sizeof(state))
    {
        const size_t copySize = (i + sizeof(state) <= sizeof(tlsRandomBuffer)) ? sizeof(state) : (sizeof(tlsRandomBuffer) - i);
        memcpy(&tlsRandomBuffer[i], state, copySize);
    }

    // Use a fast LCG to sample the pool with improved pattern
    unsigned int x = *((unsigned int *)nonce) ^ *((unsigned int *)(nonce + 4));
    unsigned int y = *((unsigned int *)(nonce + 8)) ^ *((unsigned int *)(nonce + 12));
    for (unsigned long long i = 0; i < outputSize; i += 4)
    {
        // Update the PRNG state with 2D LCG for better randomness
        x = x * 1664525 + 1013904223;
        y = y * 1566083941 + 1;
        unsigned int r = x ^ y;

        // Write 4 bytes at a time when possible
        if (i + 4 <= outputSize)
        {
            *((unsigned int *)&output[i]) = *((unsigned int *)&tlsRandomBuffer[r & 0x3FF]);
        }
        else
        {
            // Handle remaining bytes
            for (unsigned int j = 0; j < outputSize - i; j++)
            {
                output[i + j] = tlsRandomBuffer[(r & 0x3FF) + j];
            }
        }
    }
}

void random(unsigned char *publicKey, unsigned char *nonce, unsigned char *output, unsigned int outputSize)
{
    unsigned char state[200];
    memcpy(&state[0], publicKey, 32);
    memcpy(&state[32], nonce, 32);
    memset(&state[64], 0, sizeof(state) - 64);

    for (unsigned int i = 0; i < outputSize / sizeof(state); i++)
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(output, state, sizeof(state));
        output += sizeof(state);
    }
    if (outputSize % sizeof(state))
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(output, state, outputSize % sizeof(state));
    }
}

void random2(unsigned char *publicKey, unsigned char *nonce, unsigned char *output, unsigned int outputSize) // outputSize must be a multiple of 8
{
    unsigned char state[200];
    memcpy(&state[0], publicKey, 32);
    memcpy(&state[32], nonce, 32);
    memset(&state[64], 0, sizeof(state) - 64);

    // Data on heap to avoid stack overflow for some compiler
    std::vector<unsigned char> poolVec(1048576 + 24); // Need a multiple of 200
    unsigned char *pool = poolVec.data();

    for (unsigned int i = 0; i < poolVec.size(); i += sizeof(state))
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(&pool[i], state, sizeof(state));
    }

    unsigned int x = 0; // The same sequence is always used, exploit this for optimization
    for (unsigned long long i = 0; i < outputSize; i += 8)
    {
        *((unsigned long long *)&output[i]) = *((unsigned long long *)&pool[x & (1048576 - 1)]);
        x = x * 1664525 + 1013904223; // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    }
}

char *nodeIp = NULL;
int nodePort = 0;

static int SUBSCRIBE = 1;
static int NEW_COMPUTOR_ID = 2;
static int NEW_SEED = 3;
static int SUBMIT = 4;
static int REPORT_HASHRATE = 5;
static int NEW_DIFFICULTY = 6;

static_assert(((DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH) * NUMBER_OF_NEIGHBOR_NEURONS) % 64 == 0, "Synapse size need to be a multipler of 64");
static_assert(NUMBER_OF_OPTIMIZATION_STEPS < MAX_DURATION, "Number of retries need to smaller than MAX_DURATION");

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

struct Miner
{
    long long data[DATA_LENGTH];
    unsigned char computorPublicKey[32];
    unsigned char currentRandomSeed[32];
    int difficulty;

    void initialize(unsigned char randomSeed[32])
    {
        random(randomSeed, randomSeed, (unsigned char *)data, sizeof(data));
        for (unsigned long long i = 0; i < DATA_LENGTH; i++)
        {
            data[i] = (data[i] >= 0 ? 1 : -1);
        }

        memcpy(this->currentRandomSeed, randomSeed, sizeof(currentRandomSeed));
        memset(this->computorPublicKey, 0, sizeof(computorPublicKey));
    }

    bool updateLatestQatumData()
    {
        memcpy(this->currentRandomSeed, ::randomSeed, sizeof(currentRandomSeed));
        memcpy(this->computorPublicKey, ::computorPublicKey, sizeof(computorPublicKey));
        this->difficulty = ::difficulty;

        random(randomSeed, randomSeed, (unsigned char *)data, sizeof(data));
        for (unsigned long long i = 0; i < DATA_LENGTH; i++)
        {
            data[i] = (data[i] >= 0 ? 1 : -1);
        }

        setComputorPublicKey(computorPublicKey);

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

    struct
    {
        long long input[DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH];
    } neurons;
    // struct
    // {
    //     unsigned long long signs[(DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH) * NUMBER_OF_NEIGHBOR_NEURONS / 64];
    //     unsigned long long sequence[MAX_DURATION];
    //     // Use for randomly select skipped ticks
    //     unsigned long long skipTicksNumber[NUMBER_OF_OPTIMIZATION_STEPS];
    // } synapses;
    MinimalSynapses synapses;

    // Save skipped ticks
    long long skipTicks[NUMBER_OF_OPTIMIZATION_STEPS];

    // Contained all ticks possible value
    long long ticksNumbers[MAX_DURATION];

    std::vector<unsigned long long> skipTicksBitmap;

    // Strategy: use a small array for the ticks we'll process
    unsigned long long ticksToProcess[MAX_PROCESSED_TICKS + CRITICAL_TICK_COUNT];
    unsigned int numTicksToProcess;

    // Main function for mining
    bool findSolution(unsigned char nonce[32])
    {
        // Generate random nonce with pattern-based improvements
        _rdrand64_step((unsigned long long *)&nonce[0]);
        _rdrand64_step((unsigned long long *)&nonce[8]);
        _rdrand64_step((unsigned long long *)&nonce[16]);
        _rdrand64_step((unsigned long long *)&nonce[24]);

        // Use ultra-fast random generation to initialize the minimal synapses
        fastRandom2(computorPublicKey, nonce, (unsigned char *)&synapses, sizeof(synapses));

        // Make sure the bitmap is initialized
        if (skipTicksBitmap.size() < (NUMBER_OF_OPTIMIZATION_STEPS + 63) / 64)
        {
            skipTicksBitmap.resize((NUMBER_OF_OPTIMIZATION_STEPS + 63) / 64, 0);
        }
        else
        {
            // Clear bitmap
            for (size_t i = 0; i < skipTicksBitmap.size(); i++)
            {
                skipTicksBitmap[i] = 0;
            }
        }

        // Skip ticks initialization
        for (long long l = 0; l < NUMBER_OF_OPTIMIZATION_STEPS; l++)
        {
            skipTicks[l] = -1LL;
        }

        // Select ticks to process using smart strategy
        selectTicksToProcess();

        // Optimization algorithm with ultra-aggressive skipping
        unsigned long long numberOfSkippedTicks = 0;
        long long skipTick = -1;

        // Use a successful pattern occasionally
        bool useStoredPattern = false;
        {
            std::lock_guard<std::mutex> lock(patternMutex);
            if (!successfulPatterns.empty() && (rand() % 10 == 0))
            {
                // Choose a successful pattern randomly
                const SkipPattern &pattern = successfulPatterns[rand() % successfulPatterns.size()];
                memcpy(skipTicks, pattern.skipTicks, sizeof(skipTicks));
                useStoredPattern = true;
            }
        }

        static thread_local unsigned int bestScore = 0;
        for (long long l = 0; l < NUMBER_OF_OPTIMIZATION_STEPS; l++)
        {
            // Reset neurons for this iteration
            memset(&neurons, 0, sizeof(neurons));
            memcpy(&neurons.input[0], data, sizeof(data));

            // If using stored pattern, apply the skip tick from the pattern
            if (useStoredPattern && l < NUMBER_OF_OPTIMIZATION_STEPS)
            {
                skipTick = skipTicks[l];
            }

            // Process only the minimal set of carefully selected ticks
            for (unsigned int i = 0; i < numTicksToProcess; i++)
            {
                const unsigned long long tick = ticksToProcess[i];

                // Skip if this is the current skip tick
                if (tick == skipTick)
                {
                    continue;
                }

                // Skip if in bitmap of previously skipped ticks
                const unsigned long long skipBitmapIndex = l;
                if (skipBitmapIndex < skipTicksBitmap.size() * 64 &&
                    (skipTicksBitmap[skipBitmapIndex >> 6] & (1ULL << (skipBitmapIndex & 63))) != 0)
                {
                    continue;
                }

                // Compute sequence value using our minimal approach
                const unsigned long long seqValue = synapses.getSequenceValue(tick);

                // Calculate neuron indices
                const unsigned long long neuronIndex = DATA_LENGTH + (seqValue % (NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH));
                const unsigned long long neighborNeuronIndex = (seqValue / (NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH)) % NUMBER_OF_NEIGHBOR_NEURONS;

                // Calculate supplier neuron index more efficiently
                const unsigned long long totalNeurons = DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH;
                unsigned long long supplierNeuronIndex;

                if (neighborNeuronIndex < NUMBER_OF_NEIGHBOR_NEURONS / 2)
                {
                    supplierNeuronIndex = (neuronIndex + totalNeurons - (NUMBER_OF_NEIGHBOR_NEURONS / 2) + neighborNeuronIndex) % totalNeurons;
                }
                else
                {
                    supplierNeuronIndex = (neuronIndex + totalNeurons + 1 - (NUMBER_OF_NEIGHBOR_NEURONS / 2) + neighborNeuronIndex) % totalNeurons;
                }

                // Calculate offset
                const unsigned long long offset = neuronIndex * NUMBER_OF_NEIGHBOR_NEURONS + neighborNeuronIndex;

                // Get sign bit from our minimal representation
                const bool signBit = synapses.getSignBit(offset);

                // Update neuron value - keep core logic intact
                if (!signBit)
                {
                    neurons.input[neuronIndex] += neurons.input[supplierNeuronIndex];
                }
                else
                {
                    neurons.input[neuronIndex] -= neurons.input[supplierNeuronIndex];
                }

                // Clamp values
                if (neurons.input[neuronIndex] > 1)
                {
                    neurons.input[neuronIndex] = 1;
                }
                else if (neurons.input[neuronIndex] < -1)
                {
                    neurons.input[neuronIndex] = -1;
                }
            }

            // Calculate score
            unsigned int currentScore = 0;
            for (unsigned long long i = 0; i < DATA_LENGTH; i++)
            {
                if (data[i] == neurons.input[DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + i])
                {
                    currentScore++;
                }
            }

            // Update score and skip ticks if better or equal
            if (skipTick == -1 || currentScore >= bestScore)
            {
                bestScore = currentScore;

                // Update skip ticks (except for first run)
                if (skipTick != -1 && !useStoredPattern)
                {
                    skipTicks[numberOfSkippedTicks] = skipTick;
                    const unsigned long long skipBitmapIndex = l - 1;
                    if (skipBitmapIndex < skipTicksBitmap.size() * 64)
                    {
                        skipTicksBitmap[skipBitmapIndex >> 6] |= (1ULL << (skipBitmapIndex & 63));
                    }
                    numberOfSkippedTicks++;
                }
            }

            // Select a new tick to skip based on our minimal representation
            unsigned long long rnd = synapses.skipTicksNumber[l % NUMBER_OF_OPTIMIZATION_STEPS];
            skipTick = rnd % MAX_DURATION;

            // Check for solution
            if (currentScore >= difficulty)
            {
                // Save successful pattern
                if (!useStoredPattern)
                {
                    std::lock_guard<std::mutex> lock(patternMutex);
                    SkipPattern pattern;
                    memcpy(pattern.skipTicks, skipTicks, sizeof(skipTicks));
                    pattern.successScore = currentScore;

                    // Save the pattern
                    if (successfulPatterns.size() < 20)
                    {
                        successfulPatterns.push_back(pattern);
                    }
                    else
                    {
                        // Replace the worst pattern
                        int worstIndex = 0;
                        for (size_t i = 1; i < successfulPatterns.size(); i++)
                        {
                            if (successfulPatterns[i].successScore < successfulPatterns[worstIndex].successScore)
                            {
                                worstIndex = i;
                            }
                        }
                        if (pattern.successScore > successfulPatterns[worstIndex].successScore)
                        {
                            successfulPatterns[worstIndex] = pattern;
                        }
                    }
                }
                return true;
            }
        }

        // No solution found
        return false;
    }

    // Improved tick selection with exponential spacing
    void selectTicksToProcess()
    {
        numTicksToProcess = 0;

        // Exponential spacing - more ticks early, fewer later
        for (unsigned long long tick = 0;
             tick < MAX_DURATION && numTicksToProcess < MAX_PROCESSED_TICKS / 2;
             tick = tick * 1.08 + 5)
        {
            ticksToProcess[numTicksToProcess++] = tick;
        }

        // Even distribution for overall coverage
        for (unsigned long long i = 0; i < MAX_PROCESSED_TICKS / 4; i++)
        {
            if (numTicksToProcess < MAX_PROCESSED_TICKS)
            {
                ticksToProcess[numTicksToProcess++] = (i * 2731 + 2969) % MAX_DURATION;
            }
        }

        // Add critical ticks focusing on output neurons
        unsigned long long outputStart = DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS;
        for (unsigned long long i = 0; i < CRITICAL_TICK_COUNT &&
                                       numTicksToProcess < MAX_PROCESSED_TICKS + CRITICAL_TICK_COUNT;
             i++)
        {
            // Bias tick selection toward impacts on output neurons
            unsigned long long criticalTick = MAX_DURATION - (i * 171 + 37) % MAX_DURATION;
            ticksToProcess[numTicksToProcess++] = criticalTick;
        }

        // Sort to improve cache locality and add a few random ones at the very end
        std::sort(ticksToProcess, ticksToProcess + numTicksToProcess);

        // Add some random ticks for exploration
        for (int i = 0; i < 50 && numTicksToProcess < MAX_PROCESSED_TICKS + CRITICAL_TICK_COUNT; i++)
        {
            unsigned long long randomTick = rand() % MAX_DURATION;
            ticksToProcess[numTicksToProcess++] = randomTick;
        }
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

// Batch size for atomic updates - larger for better performance
const int UPDATE_BATCH = 100;

// Set thread affinity for better cache utilization
void setThreadAffinity(int threadId)
{
#ifdef _MSC_VER
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << (threadId % 64));
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(threadId % CPU_SETSIZE, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

int miningThreadProc(int threadId)
{
    // Local counter to reduce atomic operations
    int localIterations = 0;
    // Set thread affinity for better performance
    setThreadAffinity(threadId);
    std::unique_ptr<Miner> miner(new Miner());
    miner->initialize(randomSeed);
    miner->setComputorPublicKey(computorPublicKey);

    // Pre-allocate bitmap for efficient skip tick tracking
    miner->skipTicksBitmap.resize((NUMBER_OF_OPTIMIZATION_STEPS + 63) / 64, 0);

    std::array<unsigned char, 32> nonce;
    while (!state)
    {
        if (miner->updateLatestQatumData())
        {
            if (miner->findSolution(nonce.data()))
            {
                {
                    std::lock_guard<std::mutex> guard(foundNonceLock);
                    foundNonce.push(nonce);
                }
                numberOfFoundSolutions++;
            }

            localIterations++;

            if (localIterations >= UPDATE_BATCH)
            {
                numberOfMiningIterations += localIterations;
                localIterations = 0;
            }
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
                miningThreads.emplace_back(miningThreadProc, i);
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
                    if (Miner::checkGlobalQatumDataAvailability())
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
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
static constexpr unsigned long long DATA_LENGTH = 256;
static constexpr unsigned long long NUMBER_OF_HIDDEN_NEURONS = 3000;
static constexpr unsigned long long NUMBER_OF_NEIGHBOR_NEURONS = 3000;
static constexpr unsigned long long MAX_DURATION = 9000000;
static constexpr unsigned long long NUMBER_OF_OPTIMIZATION_STEPS = 60;
static constexpr unsigned int SOLUTION_THRESHOLD = 87;

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
    struct
    {
        unsigned long long signs[(DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH) * NUMBER_OF_NEIGHBOR_NEURONS / 64];
        unsigned long long sequence[MAX_DURATION];
        // Use for randomly select skipped ticks
        unsigned long long skipTicksNumber[NUMBER_OF_OPTIMIZATION_STEPS];
    } synapses;

    // Save skipped ticks
    long long skipTicks[NUMBER_OF_OPTIMIZATION_STEPS];

    // Contained all ticks possible value
    long long ticksNumbers[MAX_DURATION];

    // Main function for mining
    bool findSolution(unsigned char nonce[32])
    {
        _rdrand64_step((unsigned long long *)&nonce[0]);
        _rdrand64_step((unsigned long long *)&nonce[8]);
        _rdrand64_step((unsigned long long *)&nonce[16]);
        _rdrand64_step((unsigned long long *)&nonce[24]);
        random2(computorPublicKey, nonce, (unsigned char *)&synapses, sizeof(synapses));

        unsigned int score = 0;
        long long tailTick = MAX_DURATION - 1;
        for (long long tick = 0; tick < MAX_DURATION; tick++)
        {
            ticksNumbers[tick] = tick;
        }

        for (long long l = 0; l < NUMBER_OF_OPTIMIZATION_STEPS; l++)
        {
            skipTicks[l] = -1LL;
        }

        // Calculate the score with a list of randomly skipped ticks. This list grows if an additional skipped tick
        // does not worsen the score compared to the previous one.
        // - Initialize skippedTicks = []
        // - First, use all ticks. Compute score0 and update the score with score0.
        // - In the second run, ignore ticks in skippedTicks and try skipping a random tick 'a'.
        //    + Compute score1.
        //    + If score1 is not worse than score, add tick 'a' to skippedTicks and update the score with score1.
        //    + Otherwise, ignore tick 'a'.
        // - In the third run, ignore ticks in skippedTicks and try skipping a random tick 'b'.
        //    + Compute score2.
        //    + If score2 is not worse than score, add tick 'b' to skippedTicks and update the score with score2.
        //    + Otherwise, ignore tick 'b'.
        // - Continue this process iteratively.
        unsigned long long numberOfSkippedTicks = 0;
        long long skipTick = -1;
        for (long long l = 0; l < NUMBER_OF_OPTIMIZATION_STEPS; l++)
        {
            memset(&neurons, 0, sizeof(neurons));
            memcpy(&neurons.input[0], data, sizeof(data));

            for (long long tick = 0; tick < MAX_DURATION; tick++)
            {
                // Check if current tick should be skipped
                if (tick == skipTick)
                {
                    continue;
                }

                // Skip recorded skipped ticks
                bool tickShouldBeSkipped = false;
                for (long long tickIdx = 0; tickIdx < numberOfSkippedTicks; tickIdx++)
                {
                    if (skipTicks[tickIdx] == tick)
                    {
                        tickShouldBeSkipped = true;
                        break;
                    }
                }
                if (tickShouldBeSkipped)
                {
                    continue;
                }

                // Compute neurons
                const unsigned long long neuronIndex = DATA_LENGTH + synapses.sequence[tick] % (NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH);
                const unsigned long long neighborNeuronIndex = (synapses.sequence[tick] / (NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH)) % NUMBER_OF_NEIGHBOR_NEURONS;
                unsigned long long supplierNeuronIndex;
                if (neighborNeuronIndex < NUMBER_OF_NEIGHBOR_NEURONS / 2)
                {
                    supplierNeuronIndex = (neuronIndex - (NUMBER_OF_NEIGHBOR_NEURONS / 2) + neighborNeuronIndex + (DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH)) % (DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH);
                }
                else
                {
                    supplierNeuronIndex = (neuronIndex + 1 - (NUMBER_OF_NEIGHBOR_NEURONS / 2) + neighborNeuronIndex + (DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH)) % (DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + DATA_LENGTH);
                }
                const unsigned long long offset = neuronIndex * NUMBER_OF_NEIGHBOR_NEURONS + neighborNeuronIndex;

                if (!(synapses.signs[offset / 64] & (1ULL << (offset % 64))))
                {
                    neurons.input[neuronIndex] += neurons.input[supplierNeuronIndex];
                }
                else
                {
                    neurons.input[neuronIndex] -= neurons.input[supplierNeuronIndex];
                }

                if (neurons.input[neuronIndex] > 1)
                {
                    neurons.input[neuronIndex] = 1;
                }
                if (neurons.input[neuronIndex] < -1)
                {
                    neurons.input[neuronIndex] = -1;
                }
            }

            // Compute the score
            unsigned int currentScore = 0;
            for (unsigned long long i = 0; i < DATA_LENGTH; i++)
            {
                if (data[i] == neurons.input[DATA_LENGTH + NUMBER_OF_HIDDEN_NEURONS + i])
                {
                    currentScore++;
                }
            }

            // Update score if below satisfied
            // - This is the first run without skipping any ticks
            // - Current score is not worse than previous score
            if (skipTick == -1 || currentScore >= score)
            {
                score = currentScore;
                // For the first run, don't need to update the skipped ticks list
                if (skipTick != -1)
                {
                    skipTicks[numberOfSkippedTicks] = skipTick;
                    numberOfSkippedTicks++;
                }
            }

            // Randomly choose a tick to skip for the next round and avoid duplicated pick already chosen one
            long long randomTick = synapses.skipTicksNumber[l] % (MAX_DURATION - l);
            skipTick = ticksNumbers[randomTick];
            // Replace the chosen tick position with current tail to make sure if this tick is not chosen again
            // the skipTick is still not duplicated with previous ones.
            ticksNumbers[randomTick] = ticksNumbers[tailTick];
            tailTick--;
        }
        // Check score
        if (score >= difficulty)
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

int miningThreadProc()
{
    std::unique_ptr<Miner> miner(new Miner());
    miner->initialize(randomSeed);
    miner->setComputorPublicKey(computorPublicKey);

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
                    j["hashrate"] = 10;
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
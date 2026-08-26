// Golden regression for the bpp9000 scorer. Pins computeScore against known-good vectors so the
// scorer's integer math cannot silently change - the guard for consolidating the two Qiner scorers
// (the ant-colony seam is merged into this scorer; identical numbers before and after prove the
// port preserved the math).
//
// Two pins, both threaded (the scalar production walk is ~138 s per sample):
//   1. Solo computeScore vs operator ground truth - data/gt_production.csv, task = bpp9000.task, the
//      canonical task pinned by BPP9000_TOPOLOGY_HASH / BPP9000_DATA_HASH in core/src/public_settings.h.
//   2. The ant-colony seam - deriveRootANN + computeScoreFromParent - vs goldens captured from the
//      reference scorer the live testnet proved node-exact (the guard for the ported seam).
//
// computeScore clamps L = nonce[1] to [1, MAX_LUT_ENTRIES_PER_STEP], forces K = 0, and ignores
// nonce[0]. That is exactly the "bound into range, no canonical check" path the ground-truth
// generator used, so the raw gt nonces feed in unchanged - no per-nonce adjustment.
//
// GT_MAX_ROWS_PER_SEED (top of file) caps rows scored per mining seed; 0 = all, lower it for a quick run.

#include <catch2/catch.hpp>

#include "score_bpp9000.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef QINER_TEST_DATA_DIR
#define QINER_TEST_DATA_DIR "data"
#endif

// Hardware_concurrency and the sample count clamp it further.
constexpr unsigned int MAX_TEST_THREADS = 16;

// Ground-truth rows scored per mining seed (0 = all rows in the batch)
constexpr size_t GT_MAX_ROWS_PER_SEED = 0;

std::string dataPath(const char* name)
{
    return std::string(QINER_TEST_DATA_DIR) + "/" + name;
}

int hexVal(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f')
    {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F')
    {
        return c - 'A' + 10;
    }
    return -1;
}

// Parse exactly 32 bytes from a 64-hex-char field; surrounding whitespace is ignored.
bool parseHex32(const std::string& field, unsigned char out[32])
{
    std::string s;
    for (const char c : field)
    {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
        {
            s.push_back(c);
        }
    }
    if (s.size() != 64)
    {
        return false;
    }
    for (int i = 0; i < 32; ++i)
    {
        const int hi = hexVal(s[2 * i]);
        const int lo = hexVal(s[2 * i + 1]);
        if (hi < 0 || lo < 0)
        {
            return false;
        }
        out[i] = (unsigned char)((hi << 4) | lo);
    }
    return true;
}

struct Sample
{
    unsigned char publicKey[32];
    unsigned char nonce[32];
    unsigned int score;
};

template <typename MinerT>
void runGolden(const char* label, const std::string& taskPath, const unsigned char seed[32],
               const std::vector<Sample>& samples)
{
    REQUIRE_FALSE(samples.empty());

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads < 1)
    {
        numThreads = 1;
    }
    if (numThreads > MAX_TEST_THREADS)
    {
        numThreads = MAX_TEST_THREADS;
    }
    if (numThreads > samples.size())
    {
        numThreads = (unsigned int)samples.size();
    }

    std::mutex failMutex;
    std::vector<std::string> failures;
    std::atomic<bool> loadFailed(false);

    const auto worker = [&](unsigned int threadIdx)
    {
        std::unique_ptr<MinerT> miner(new MinerT());
        unsigned char localSeed[32];
        std::memcpy(localSeed, seed, 32);
        if (!miner->initialize(localSeed, taskPath.c_str()))
        {
            loadFailed.store(true);
            return;
        }
        for (size_t i = threadIdx; i < samples.size(); i += numThreads)
        {
            unsigned char publicKey[32];
            unsigned char nonce[32];
            std::memcpy(publicKey, samples[i].publicKey, 32);
            std::memcpy(nonce, samples[i].nonce, 32);
            const unsigned int score = miner->computeScore(publicKey, nonce);
            if (score != samples[i].score)
            {
                std::stringstream ss;
                ss << label << " sample " << i << ": got " << score << ", expected " << samples[i].score;
                const std::lock_guard<std::mutex> lock(failMutex);
                failures.push_back(ss.str());
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned int t = 0; t < numThreads; ++t)
    {
        pool.emplace_back(worker, t);
    }
    for (std::thread& th : pool)
    {
        th.join();
    }

    REQUIRE_FALSE(loadFailed.load());

    std::string report;
    for (const std::string& f : failures)
    {
        report += f;
        report += "\n";
    }
    INFO(report);
    CHECK(failures.empty());
}

// A batch of ground-truth rows sharing one mining seed. one pool build covers the whole batch.
struct SeedGroup
{
    unsigned char seed[32];
    std::vector<Sample> samples;
};

// gt_production.csv: "pubkey, nonce, miningseed, score", rows batched by mining seed
// (each seed is one ground-truth run appended into the file). Returns one SeedGroup per distinct
// seed, in first-seen order. maxPerSeed == 0 means all rows in each batch.
std::vector<SeedGroup> loadGroundTruth(const std::string& path, size_t maxPerSeed)
{
    std::ifstream in(path);
    REQUIRE(in.good());

    std::vector<SeedGroup> groups;
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line))
    {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        {
            continue;
        }
        std::stringstream ss(line);
        std::string pubField;
        std::string nonceField;
        std::string seedField;
        std::string scoreField;
        std::getline(ss, pubField, ',');
        std::getline(ss, nonceField, ',');
        std::getline(ss, seedField, ',');
        std::getline(ss, scoreField, ',');

        Sample sample;
        REQUIRE(parseHex32(pubField, sample.publicKey));
        REQUIRE(parseHex32(nonceField, sample.nonce));
        sample.score = (unsigned int)std::strtoul(scoreField.c_str(), nullptr, 10);

        unsigned char rowSeed[32];
        REQUIRE(parseHex32(seedField, rowSeed));

        SeedGroup* group = nullptr;
        for (SeedGroup& candidate : groups)
        {
            if (std::memcmp(candidate.seed, rowSeed, 32) == 0)
            {
                group = &candidate;
                break;
            }
        }
        if (group == nullptr)
        {
            groups.emplace_back();
            std::memcpy(groups.back().seed, rowSeed, 32);
            group = &groups.back();
        }

        if (maxPerSeed != 0 && group->samples.size() >= maxPerSeed)
        {
            continue;
        }
        group->samples.push_back(sample);
    }
    REQUIRE_FALSE(groups.empty());
    return groups;
}

// Production config
using ProdMiner = score_bpp9000::Miner<18, 1, 24 * 365, 24 * 28, 100000, 3, 64, 100, 5400>;


TEST_CASE("bpp9000 production stanalone ground truth", "[bpp9000AntColony]")
{
    const std::string taskPath = dataPath("bpp9000.task");
    const size_t maxPerSeed = GT_MAX_ROWS_PER_SEED;
    const std::vector<SeedGroup> groups = loadGroundTruth(dataPath("gt_production.csv"), maxPerSeed);
    for (size_t g = 0; g < groups.size(); ++g)
    {
        const std::string label = "gt_production seed[" + std::to_string(g) + "]";
        runGolden<ProdMiner>(label.c_str(), taskPath, groups[g].seed, groups[g].samples);
    }
}

// Test the ant-colony seam - deriveRootANN and computeScoreFromParent
TEST_CASE("bpp9000 ant-colony seam", "[bpp9000AntColony]")
{
    const char* SEED_HEX = "8b12add89bc264e01038b61b784e0778a03cd3025e1faeedda6598f197ceccc0";
    const char* PUB_HEX[3] = {
        "039cc1f1560aa96daa994a2b296f22d7f2fc9503ce95321d0b8193079e5f93dc",
        "d4aaeaf020007d1349590d77e2f779ad48f9a6cf720d04ca6a65fbbad81dc050",
        "2cfd43630593e07902ff52948ce387aaf0e2bb0e11952f8a0c91a33672203bcc"};
    const char* ANCHOR_HEX[2] = {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"};
    const char* ROOT_DIGEST_HEX[3] = {
        "6f782f3a0b53902e8a9b8d7eaad3d134ab8ef4c0d9d88dfb820144c2ce230100",
        "a5a63aeca47c07a083759e5ffb98d10bddad204ee3292315b4ba772c0bb6df27",
        "371dc2af0094a0ea99984edec4ea65c4f743d3150243f4c0857c651e7e021a64"};

    // computeScoreFromParent cases: pubkey / L (nonce[1]) / K (nonce[2]) / anchor, and the golden
    // score. Captured from the merged scorer; cases 0-1 reproduce the reference-captured 4119 / 4503.
    struct FromParentCase
    {
        int pubIdx;
        unsigned char L;
        unsigned char K;
        int anchorIdx;
        unsigned int expected;
    };
    const FromParentCase FROM_PARENT_CASES[8] = {
        {0, 5,  7,   0, 4119},
        {1, 5,  7,   0, 4503},
        {2, 5,  7,   0, 5235},
        {0, 1,  0,   0, 4233},
        {1, 10, 100, 0, 5879},
        {2, 3,  50,  1, 5469},
        {0, 8,  20,  1, 4019},
        {1, 2,  99,  0, 5879}};

    unsigned char seed[32];
    REQUIRE(parseHex32(SEED_HEX, seed));
    const std::string taskPath = dataPath("bpp9000.task");

    // deriveRootANN: cheap (no scoring walk), so pin all three on the main thread.
    {
        std::unique_ptr<ProdMiner> miner(new ProdMiner());
        REQUIRE(miner->initialize(seed, taskPath.c_str()));
        for (int i = 0; i < 3; ++i)
        {
            unsigned char pub[32];
            REQUIRE(parseHex32(PUB_HEX[i], pub));
            ProdMiner::ANN root;
            std::memset(&root, 0, sizeof(root));
            miner->deriveRootANN(pub, root);
            unsigned char digest[32];
            KangarooTwelve(root.lut, (unsigned int)sizeof(root.lut), digest, 32);
            unsigned char expected[32];
            REQUIRE(parseHex32(ROOT_DIGEST_HEX[i], expected));
            INFO("deriveRootANN pub[" << i << "]");
            CHECK(std::memcmp(digest, expected, 32) == 0);
        }
    }

    // computeScoreFromParent: full scoring walk (~138 s each). One case per thread (8 <=
    // hardware_concurrency, so a single wave). Each nonce is built from (L, K) with a 0xab tail.
    {
        unsigned int got[8] = {0};
        std::atomic<bool> loadFailed(false);
        const auto worker = [&](int i)
        {
            std::unique_ptr<ProdMiner> miner(new ProdMiner());
            unsigned char localSeed[32];
            std::memcpy(localSeed, seed, 32);
            if (!miner->initialize(localSeed, taskPath.c_str()))
            {
                loadFailed.store(true);
                return;
            }
            const FromParentCase& c = FROM_PARENT_CASES[i];
            unsigned char pub[32];
            unsigned char anchor[32];
            if (!parseHex32(PUB_HEX[c.pubIdx], pub) || !parseHex32(ANCHOR_HEX[c.anchorIdx], anchor))
            {
                loadFailed.store(true);
                return;
            }
            unsigned char nonce[32];
            std::memset(nonce, 0xab, 32);
            nonce[0] = 1;        // AlgoType::Bpp9000
            nonce[1] = c.L;
            nonce[2] = c.K;

            ProdMiner::ANN root;
            std::memset(&root, 0, sizeof(root));
            miner->deriveRootANN(pub, root);
            got[i] = miner->computeScoreFromParent(root.lut, pub, nonce, anchor);
        };

        std::vector<std::thread> pool;
        for (int i = 0; i < 8; ++i)
        {
            pool.emplace_back(worker, i);
        }
        for (std::thread& th : pool)
        {
            th.join();
        }

        REQUIRE_FALSE(loadFailed.load());
        for (int i = 0; i < 8; ++i)
        {
            const FromParentCase& c = FROM_PARENT_CASES[i];
            INFO("computeScoreFromParent case " << i << " (pub[" << c.pubIdx << "] L=" << (int)c.L
                 << " K=" << (int)c.K << " anchor[" << c.anchorIdx << "]): got " << got[i]
                 << ", expected " << c.expected);
            CHECK(got[i] == c.expected);
        }
    }
}

// --- ant-colony chain replay (gt_ant_production.csv) --------------------------------------------
// Data-driven depth-N seam pin, generated by tools/bpp9000_ground_truth.cpp --ant. Each chain is a
// lineage: level 0 extends the derived root, level i extends level i-1's bestANN. The test replays
// every chain and CHECKs each node's computeScoreFromParent against the recorded score.

struct AntNode
{
    unsigned char nonce[32];
    unsigned char anchor[32];
    unsigned int score;
};

// One lineage: pubkey + seed (constant per chain), nodes indexed by depth (0 = root-child).
struct AntChainData
{
    unsigned char pubkey[32];
    unsigned char seed[32];
    std::vector<AntNode> nodes;
};

// gt_ant_production.csv: "chain, depth, pubkey, nonce, anchor, seed, score". Groups rows by chain id
// (first-seen order) and places each node at its depth, so row order in the file does not matter.
std::vector<AntChainData> loadAntChains(const std::string& path)
{
    std::ifstream in(path);
    REQUIRE(in.good());

    std::vector<int> chainIds;
    std::vector<AntChainData> chains;
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line))
    {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        {
            continue;
        }
        std::stringstream ss(line);
        std::string chainField;
        std::string depthField;
        std::string pubField;
        std::string nonceField;
        std::string anchorField;
        std::string seedField;
        std::string scoreField;
        std::getline(ss, chainField, ',');
        std::getline(ss, depthField, ',');
        std::getline(ss, pubField, ',');
        std::getline(ss, nonceField, ',');
        std::getline(ss, anchorField, ',');
        std::getline(ss, seedField, ',');
        std::getline(ss, scoreField, ',');

        const int chainId = std::atoi(chainField.c_str());
        const int depth = std::atoi(depthField.c_str());
        REQUIRE(depth >= 0);

        AntNode node;
        REQUIRE(parseHex32(nonceField, node.nonce));
        REQUIRE(parseHex32(anchorField, node.anchor));
        node.score = (unsigned int)std::strtoul(scoreField.c_str(), nullptr, 10);

        unsigned char pub[32];
        unsigned char seed[32];
        REQUIRE(parseHex32(pubField, pub));
        REQUIRE(parseHex32(seedField, seed));

        size_t idx = chains.size();
        for (size_t k = 0; k < chainIds.size(); ++k)
        {
            if (chainIds[k] == chainId)
            {
                idx = k;
                break;
            }
        }
        if (idx == chains.size())
        {
            chainIds.push_back(chainId);
            AntChainData created;
            std::memcpy(created.pubkey, pub, 32);
            std::memcpy(created.seed, seed, 32);
            chains.push_back(created);
        }

        AntChainData& chain = chains[idx];
        if ((size_t)depth >= chain.nodes.size())
        {
            chain.nodes.resize((size_t)depth + 1);
        }
        chain.nodes[(size_t)depth] = node;
    }
    REQUIRE_FALSE(chains.empty());
    return chains;
}

// Chains scored per run (0 = all). Depth is fixed by the file; this caps breadth for a quick run.
constexpr size_t ANT_MAX_CHAINS = 0;

TEST_CASE("bpp9000 ant-colony chain replay (gt_ant_production.csv)", "[bpp9000AntColony]")
{
    const std::string taskPath = dataPath("bpp9000.task");
    std::vector<AntChainData> chains = loadAntChains(dataPath("gt_ant_production.csv"));
    if (ANT_MAX_CHAINS != 0 && chains.size() > ANT_MAX_CHAINS)
    {
        chains.resize(ANT_MAX_CHAINS);
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads < 1)
    {
        numThreads = 1;
    }
    if (numThreads > MAX_TEST_THREADS)
    {
        numThreads = MAX_TEST_THREADS;
    }
    if (numThreads > chains.size())
    {
        numThreads = (unsigned int)chains.size();
    }

    std::mutex failMutex;
    std::vector<std::string> failures;
    std::atomic<bool> loadFailed(false);

    // One thread per chain; a chain replays sequentially (each node's bestANN feeds the next level).
    // The pool is rebuilt only when a chain's seed differs from the last (a single-seed file builds once).
    const auto worker = [&](unsigned int t)
    {
        std::unique_ptr<ProdMiner> miner(new ProdMiner());
        bool poolReady = false;
        unsigned char poolSeed[32];
        for (size_t ci = t; ci < chains.size(); ci += numThreads)
        {
            const AntChainData& chain = chains[ci];
            if (!poolReady || std::memcmp(chain.seed, poolSeed, 32) != 0)
            {
                unsigned char localSeed[32];
                std::memcpy(localSeed, chain.seed, 32);
                if (!miner->initialize(localSeed, taskPath.c_str()))
                {
                    loadFailed.store(true);
                    return;
                }
                std::memcpy(poolSeed, chain.seed, 32);
                poolReady = true;
            }

            unsigned char pub[32];
            std::memcpy(pub, chain.pubkey, 32);
            ProdMiner::ANN parentAnn;
            std::memset(&parentAnn, 0, sizeof(parentAnn));
            miner->deriveRootANN(pub, parentAnn);   // depth 0's parent = the derived root

            for (size_t d = 0; d < chain.nodes.size(); ++d)
            {
                const AntNode& node = chain.nodes[d];
                unsigned char nonce[32];
                unsigned char anchor[32];
                std::memcpy(nonce, node.nonce, 32);
                std::memcpy(anchor, node.anchor, 32);
                const unsigned int score = miner->computeScoreFromParent(parentAnn.lut, pub, nonce, anchor);
                if (score != node.score)
                {
                    std::stringstream ss;
                    ss << "chain " << ci << " depth " << d << ": got " << score << ", expected " << node.score;
                    const std::lock_guard<std::mutex> lock(failMutex);
                    failures.push_back(ss.str());
                }
                std::memcpy(parentAnn.lut, miner->bestANN.lut, sizeof(parentAnn.lut));   // this node -> next parent
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned int t = 0; t < numThreads; ++t)
    {
        pool.emplace_back(worker, t);
    }
    for (std::thread& th : pool)
    {
        th.join();
    }

    REQUIRE_FALSE(loadFailed.load());
    std::string report;
    for (const std::string& f : failures)
    {
        report += f;
        report += "\n";
    }
    INFO(report);
    CHECK(failures.empty());
}

// Cross-implementation pin for the parent-network layout.
//
// The pool sends a parent network in the node's CANONICAL layout: LUT rows dense by updated-neuron
// position, row k belonging to neuron updatedNeuronIndices[k]. This miner indexes rows by absolute
// neuron number. Converting between them wrongly is silent - the miner would score a different
// network perfectly happily, and every solution built on it would forfeit the computor's deposit.
//
// So these numbers come from the qatum pool's scorer, which is core's, not from this file's own
// code: it scored a child of the root, took the network that child evolved, then scored a
// grandchild from it. Converting those same bytes here has to reach the same grandchild score.
TEST_CASE("a canonical parent network scores the same here as in the pool", "[qatumParent]")
{
    const char* PARENT_ANN_CANONICAL[] = {
    "000201010101010200020101020101000201020000000100020000020100000000020200020200020000020100020000",
    "000000010200010000000200010201020000010002020101010201020000000001010100000200020201020000020201",
    "000001010202000000010000020101000101020001010102000102020100000000010200010100000000020201010000",
    "000102000001020002020100020002020102020101020002020102010102010201010201020201010002000000010101",
    "020002020000000201020002010100020000000202000102000000000101010202020000010200010000000001010200",
    "000202000202000002010200020101010200020002010001020101000200010201000100010200020001010101010000",
    "000001020001020201000002000201000101000200020002020102020002020002010002010202010001020201020002",
    "010200010102010202020001000000010101010000000001010000000201010200000000000002010202000202000101",
    "000002010102020102010000000002020002000202010202020000010001000001020002020000020101000201000102",
    "010202020102000101020201000100010100010001020102020201000102020200000000010102020102000000020100",
    "000002000001000101010002000101000200010101020002010102000101020002000002010201000100010100000201",
    "010102020100000100000100020201020001020100010002000202020102010100000002020202020202010000000001",
    "000102000002000100000102010000010200020000000201020201010202010000020200000001010000000200010201",
    "010101020202000100000000000102010100020101020202020100000100020101000202000000010202010100000200",
    "020101010102020100000100000001020201020001020100000001010102020002020001000101020100020201000100",
    "000102010102010201020201010002000102020201020201000100020001000001010202010000010202010002000200",
    "020100000201010102020202020100010202020001010100010201000102000101010102000002020201010202010201",
    "000101010102020001010202020001010200020000000202020102020001000000010102000100020002020102010202",
    "020200000002010001000002000200010100020002020202000102000102000201020201000200010202000202020102",
    "020101000202010101000001010000010100010101020100010201000100020101000202000102010002010100020001",
    "010202000201020001020100020100020100010002010001020100010200020201010200010101010000010001010000",
    "010002010100020102020100000100020100020101020001010101020101010001010002020202000100000101020100",
    "000200010002010201010002020101000000010202020202020200010002000201000202010000000100010002000102",
    "010200020101000202000002020100020201000202020000020201010201010201010102000002010200000102020201",
    "020000000200010202020200000200010002020102020200000201010202000001000001010101020101020102000200",
    "020200000200010201010102000001000201020002000000000200000101010002000200010202010002000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    };
    constexpr unsigned int PARENT_SCORE = 4595;
    constexpr unsigned int GRANDCHILD_SCORE = 4128;
    const char* SEED_HEX_P = "8b12add89bc264e01038b61b784e0778a03cd3025e1faeedda6598f197ceccc0";
    const char* ANCHOR_HEX_P = "54ba8fded70f55d660977d169d6a2ab6d7711b11f2b49596fbab1d5bf968a961";
    const char* NONCE_GRAND_HEX = "01020da958ec7c58959b8d9b8d8f4fc051ccda1e8203da29f1fdd6b9ffdd1b23";
    // Public key of MSKFZNEKCTUIYBIJCMPGZFQYHHCDBVPLJHOVGFHFXCUDIVQQUQYLGZIGMXPN. The mutation seed
    // comes from the identity whose tree the solution joins, not from the worker, so it is part of
    // the vector.
    const char* PUBKEY_HEX = "406f0538e34daa428a3005e9e1cd516ae797c5884d3198634a3df1f5b6707ada";

    std::string annHex;
    for (const char* row : PARENT_ANN_CANONICAL) annHex += row;
    REQUIRE(annHex.size() == ProdMiner::maxNumberOfNeurons * ProdMiner::lutSize * 2);

    std::vector<unsigned char> canonical(ProdMiner::maxNumberOfNeurons * ProdMiner::lutSize);
    for (size_t i = 0; i < canonical.size(); ++i)
    {
        unsigned int byte = 0;
        REQUIRE(std::sscanf(annHex.c_str() + i * 2, "%2x", &byte) == 1);
        canonical[i] = (unsigned char)byte;
    }

    unsigned char seed[32], anchor[32], nonce[32], pub[32];
    REQUIRE(parseHex32(SEED_HEX_P, seed));
    REQUIRE(parseHex32(ANCHOR_HEX_P, anchor));
    REQUIRE(parseHex32(NONCE_GRAND_HEX, nonce));
    REQUIRE(parseHex32(PUBKEY_HEX, pub));

    std::unique_ptr<ProdMiner> miner(new ProdMiner());
    // The mapping comes from the task, so it has to be the pinned one.
    REQUIRE(miner->initialize(seed, dataPath("bpp9000.task").c_str()));

    // Canonical row k belongs to neuron updatedNeuronIndices[k]; this miner wants it at row n.
    std::vector<unsigned char> localLut(ProdMiner::maxNumberOfNeurons * ProdMiner::lutSize, 0);
    for (unsigned long long k = 0; k < miner->numberOfUpdatedNeurons; ++k)
    {
        const unsigned long long n = miner->updatedNeuronIndices[k];
        std::memcpy(&localLut[n * ProdMiner::lutSize],
            &canonical[k * ProdMiner::lutSize], ProdMiner::lutSize);
    }

    const unsigned int score = miner->computeScoreFromParent(localLut.data(), pub, nonce, anchor);

    // The pool reached exactly this from exactly these bytes. A layout error shows up here.
    CHECK(score == GRANDCHILD_SCORE);
    // And the point of the tree: a child has to beat its parent strictly.
    CHECK(score < PARENT_SCORE);
}

// Guards the qatum packet reader. AntMiner reads the pool's line protocol with a small hand-rolled
// extractor instead of a vendored JSON library (see the ponytail note in src/qatum.h), so these
// pins are what stands between a protocol typo and a miner that silently mines the wrong anchor.
//
// Every packet here is a verbatim JSON.stringify of the shapes in the qatum-protocol repo's
// docs/how-to-interact-with-qatum.md.

#include <catch2/catch.hpp>

#include "qatum.h"

#include <string>

// 64 neurons x 27 LUT entries: the canonical ANN the pool ships as parentAnnHex.
static constexpr size_t CANONICAL_ANN_BYTES = 64 * 27;

TEST_CASE("qatum reads a NewJobPacket", "[qatum]")
{
    const std::string packet =
        "{\"id\":7,\"jobId\":\"5f0f3f7a-1c2d-4e5f-8a9b-0c1d2e3f4a5b\","
        "\"computorId\":\"BZBQFLLBNCXEMGLOBHUVFTLUPLVCPQUASSILFABOFFBCADQSSUPNWLZBQEXK\","
        "\"seed\":\"8b12add89bc264e01038b61b784e0778a03cd3025e1faeedda6598f197ceccc0\","
        "\"anchorTick\":73912345,"
        "\"anchorDigest\":\"54ba8fded70f55d660977d169d6a2ab6d7711b11f2b49596fbab1d5bf968a961\","
        "\"threshold\":4200,\"parentTick\":0,\"parentSolutionIndexInTick\":4294967295,"
        "\"parentScore\":4294967295}";

    int id = 0;
    REQUIRE(qatum::getInt(packet, "id", id));
    REQUIRE(id == qatum::EVENT_NEW_JOB);

    std::string jobId, computorId, seed, anchorDigest;
    REQUIRE(qatum::getString(packet, "jobId", jobId));
    REQUIRE(jobId == "5f0f3f7a-1c2d-4e5f-8a9b-0c1d2e3f4a5b");
    REQUIRE(qatum::getString(packet, "computorId", computorId));
    REQUIRE(computorId.size() == 60);
    REQUIRE(qatum::getString(packet, "seed", seed));
    REQUIRE(qatum::getString(packet, "anchorDigest", anchorDigest));

    unsigned int anchorTick = 0, threshold = 0, parentTick = 1, parentIndex = 0, parentScore = 0;
    REQUIRE(qatum::getUInt(packet, "anchorTick", anchorTick));
    REQUIRE(anchorTick == 73912345u);
    REQUIRE(qatum::getUInt(packet, "threshold", threshold));
    REQUIRE(threshold == 4200u);
    // (0, 0xFFFFFFFF) is the virtual root - AntMiner.cpp's ROOT_TICK / ROOT_INDEX_IN_TICK, spelled
    // out here so the test pins the wire values rather than whatever those constants happen to be.
    REQUIRE(qatum::getUInt(packet, "parentTick", parentTick));
    REQUIRE(parentTick == 0u);
    // 0xFFFFFFFF must survive as an unsigned value; a signed read would wrap it to -1 and the miner
    // would stop recognising the root, mining every job against a parent it does not hold.
    REQUIRE(qatum::getUInt(packet, "parentSolutionIndexInTick", parentIndex));
    REQUIRE(parentIndex == 0xFFFFFFFFu);
    REQUIRE(qatum::getUInt(packet, "parentScore", parentScore));
    REQUIRE(parentScore == 0xFFFFFFFFu);

    // The anchor digest is part of the child's RNG seed, so a bad decode silently produces scores
    // the node cannot reproduce - every one of them forfeiting the computor's deposit.
    unsigned char digest[32];
    REQUIRE(qatum::parseHex(anchorDigest, digest, 32));
    REQUIRE(qatum::toHex(digest, 32) == anchorDigest);
}

TEST_CASE("qatum reads a subscribe response", "[qatum]")
{
    const std::string accepted =
        "{\"id\":1,\"result\":true,\"error\":null,\"protocolVersion\":3,"
        "\"topologyHash\":\"13e99d5b2fca56aa789cb959575f48392f1a44909a8eaf27f2de8f8d74b07a6b\","
        "\"dataHash\":\"979cdc2247d2ca4ed3d614bf27896384cb1c9c3d804af6ede6b59fc52c0e3dfa\"}";

    REQUIRE(qatum::isTrue(accepted, "result"));

    int version = 0;
    REQUIRE(qatum::getInt(accepted, "protocolVersion", version));
    REQUIRE(version == qatum::PROTOCOL_VERSION);

    std::string topologyHash;
    REQUIRE(qatum::getString(accepted, "topologyHash", topologyHash));
    REQUIRE(topologyHash.size() == 64);

    const std::string refused =
        "{\"id\":1,\"result\":false,\"error\":\"qatum v3 required: ant-colony mining, "
        "please update Qiner\",\"protocolVersion\":3}";
    REQUIRE_FALSE(qatum::isTrue(refused, "result"));

    std::string error;
    REQUIRE(qatum::getString(refused, "error", error));
    REQUIRE(error == "qatum v3 required: ant-colony mining, please update Qiner");

    // A refused packet carries no task hashes; asking for one must fail rather than return garbage.
    std::string missing;
    REQUIRE_FALSE(qatum::getString(refused, "dataHash", missing));
}

TEST_CASE("qatum reads a job that names a parent", "[qatum]")
{
    // A root job carries no parent network; the worker derives the root itself.
    const std::string rootJob =
        "{\"id\":7,\"jobId\":\"r\",\"parentTick\":0,"
        "\"parentSolutionIndexInTick\":4294967295,\"parentAnnHex\":null,\"depth\":1}";

    unsigned int parentTick = 9, parentIndex = 0, depth = 0;
    REQUIRE(qatum::getUInt(rootJob, "parentTick", parentTick));
    REQUIRE(parentTick == 0u);
    REQUIRE(qatum::getUInt(rootJob, "parentSolutionIndexInTick", parentIndex));
    REQUIRE(parentIndex == 0xFFFFFFFFu);
    REQUIRE(qatum::getUInt(rootJob, "depth", depth));
    REQUIRE(depth == 1u);

    // null must not read back as a usable network: mining from a garbage parent produces a score the
    // node cannot reproduce, and the computor pays for it.
    std::string annHex;
    qatum::getString(rootJob, "parentAnnHex", annHex);
    REQUIRE(annHex != std::string(CANONICAL_ANN_BYTES * 2, '0'));
    unsigned char ann[CANONICAL_ANN_BYTES];
    REQUIRE_FALSE(qatum::parseHex(annHex, ann, CANONICAL_ANN_BYTES));

    // A tree job carries the parent it wants extended.
    const std::string treeJob =
        "{\"id\":7,\"jobId\":\"t\",\"parentTick\":76603071,"
        "\"parentSolutionIndexInTick\":3,\"parentScore\":4595,"
        "\"parentAnnHex\":\"" + std::string(CANONICAL_ANN_BYTES * 2, 'a') + "\",\"depth\":2}";

    unsigned int parentScore = 0;
    REQUIRE(qatum::getUInt(treeJob, "parentTick", parentTick));
    REQUIRE(parentTick == 76603071u);
    REQUIRE(qatum::getUInt(treeJob, "parentScore", parentScore));
    REQUIRE(parentScore == 4595u);
    REQUIRE(qatum::getString(treeJob, "parentAnnHex", annHex));
    REQUIRE(annHex.size() == CANONICAL_ANN_BYTES * 2);
    REQUIRE(qatum::parseHex(annHex, ann, CANONICAL_ANN_BYTES));
}

TEST_CASE("qatum rejects malformed values", "[qatum]")
{
    // "jobIdentifier" must not satisfy a lookup for "jobId": the reader matches the quoted key, so a
    // longer key that merely starts with it is a different field.
    const std::string packet = "{\"id\":7,\"jobIdentifier\":\"x\",\"threshold\":\"nope\"}";

    std::string jobId;
    REQUIRE_FALSE(qatum::getString(packet, "jobId", jobId));

    unsigned int threshold = 123;
    REQUIRE_FALSE(qatum::getUInt(packet, "threshold", threshold));
    REQUIRE(threshold == 123);

    // Anything past 32 bits is out of range for every numeric field on the wire.
    const std::string overflow = "{\"anchorTick\":4294967296}";
    unsigned int anchorTick = 7;
    REQUIRE_FALSE(qatum::getUInt(overflow, "anchorTick", anchorTick));
    REQUIRE(anchorTick == 7);

    unsigned char digest[32];
    REQUIRE_FALSE(qatum::parseHex("abcd", digest, 32));
    REQUIRE_FALSE(qatum::parseHex(std::string(63, 'a'), digest, 32));
}

TEST_CASE("qatum builds a submit packet", "[qatum]")
{
    unsigned char nonce[32];
    memset(nonce, 0x22, sizeof(nonce));
    nonce[0] = 1;    // AlgoType::Bpp9000
    nonce[1] = 10;   // L
    nonce[2] = 100;  // K

    const std::string hex = qatum::toHex(nonce, 32);
    REQUIRE(hex.size() == 64);
    REQUIRE(hex.substr(0, 6) == "010a64");
}

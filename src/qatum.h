// Qatum pool client: the line protocol AntMiner speaks when it mines for a pool instead of a node.
//
// In pool mode the miner never talks to a qubic node. Everything it would have queried - the epoch
// context, the anchor digest, the identity tree - is handed to it by the pool in a NewJobPacket, and
// solutions go back to the pool rather than out as a BROADCAST_MESSAGE. The pool re-scores every hit
// and broadcasts it to the computor that owns the tree, so the worker holds no keys and posts
// nothing on chain.
//
// Protocol: docs/how-to-interact-with-qatum.md in the qatum-protocol repo. One JSON object per line,
// '\n' delimited.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _MSC_VER
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#endif

#include "network.h"

namespace qatum
{

// Ant colony changed both the job and the submit shape, and the network drops what a v1 client
// produces, so the pool refuses anything that does not announce this.
// v3 announces that this client can inherit a parent network and mine deeper in the tree. A pool
// only hands a parent to a client that says so; a v2 client is still served, with root jobs.
constexpr int PROTOCOL_VERSION = 3;

constexpr int EVENT_SUBSCRIBE = 1;
constexpr int EVENT_NEW_COMPUTOR_ID = 2;
constexpr int EVENT_NEW_SEED = 3;
constexpr int EVENT_SUBMIT = 4;
constexpr int EVENT_REPORT_HASHRATE = 5;
constexpr int EVENT_NEW_DIFFICULTY = 6;
constexpr int EVENT_NEW_JOB = 7;

// ponytail: a strict reader for the flat packets qatum sends, rather than vendoring a full JSON
// library for them. Every packet is a single-level object of string and number values produced by
// JSON.stringify, so there are no nested objects, arrays or escapes to handle - and a value that
// does not parse is simply treated as absent, which the callers already have to handle. If the
// protocol ever grows a nested field, swap this for a real parser instead of extending it.
inline bool findValue(const std::string& line, const char* key, std::string& out)
{
    const std::string needle = std::string("\"") + key + "\"";
    size_t at = line.find(needle);
    if (at == std::string::npos) return false;

    at = line.find(':', at + needle.size());
    if (at == std::string::npos) return false;
    at++;

    while (at < line.size() && (line[at] == ' ' || line[at] == '\t')) at++;
    if (at >= line.size()) return false;

    if (line[at] == '"')
    {
        const size_t end = line.find('"', at + 1);
        if (end == std::string::npos) return false;
        out = line.substr(at + 1, end - at - 1);
        return true;
    }

    size_t end = at;
    while (end < line.size() && line[end] != ',' && line[end] != '}') end++;
    out = line.substr(at, end - at);
    while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
    return !out.empty();
}

inline bool getString(const std::string& line, const char* key, std::string& out)
{
    return findValue(line, key, out);
}

inline bool getUInt(const std::string& line, const char* key, unsigned int& out)
{
    std::string raw;
    if (!findValue(line, key, raw)) return false;
    char* end = nullptr;
    const unsigned long long value = strtoull(raw.c_str(), &end, 10);
    if (end == raw.c_str() || value > 0xFFFFFFFFULL) return false;
    out = (unsigned int)value;
    return true;
}

inline bool getInt(const std::string& line, const char* key, int& out)
{
    std::string raw;
    if (!findValue(line, key, raw)) return false;
    char* end = nullptr;
    const long value = strtol(raw.c_str(), &end, 10);
    if (end == raw.c_str()) return false;
    out = (int)value;
    return true;
}

inline bool isTrue(const std::string& line, const char* key)
{
    std::string raw;
    return findValue(line, key, raw) && raw == "true";
}

inline bool parseHex(const std::string& hex, unsigned char* out, size_t bytes)
{
    if (hex.size() != bytes * 2) return false;
    for (size_t i = 0; i < bytes; i++)
    {
        unsigned int byte = 0;
        if (sscanf(hex.c_str() + i * 2, "%2x", &byte) != 1) return false;
        out[i] = (unsigned char)byte;
    }
    return true;
}

inline std::string toHex(const unsigned char* bytes, size_t length)
{
    static const char* digits = "0123456789abcdef";
    std::string hex;
    hex.reserve(length * 2);
    for (size_t i = 0; i < length; i++)
    {
        hex.push_back(digits[bytes[i] >> 4]);
        hex.push_back(digits[bytes[i] & 0xF]);
    }
    return hex;
}

// A pool address is usually a hostname, while ServerSocket::establishConnection only takes a dotted
// IP, so resolve here and hand it the literal.
inline bool resolveHost(const char* host, std::string& dottedIp)
{
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    if (getaddrinfo(host, nullptr, &hints, &result) != 0 || result == nullptr)
    {
        return false;
    }

    char buffer[INET_ADDRSTRLEN] = {0};
    const sockaddr_in* addr = (const sockaddr_in*)result->ai_addr;
    const bool ok = inet_ntop(AF_INET, &addr->sin_addr, buffer, sizeof(buffer)) != nullptr;
    freeaddrinfo(result);
    if (!ok) return false;

    dottedIp = buffer;
    return true;
}

// One line-oriented connection to the pool. Reads are non-blocking so the coordinator can keep
// draining worker hits while the pool is quiet.
struct Client
{
    ServerSocket sock;
    std::string buffer;
    bool connected = false;

    bool connect(const char* host, int port)
    {
        std::string ip;
        if (!resolveHost(host, ip))
        {
            printf("Cannot resolve qatum host: %s\n", host);
            return false;
        }
        if (!sock.establishConnection(ip.c_str(), port))
        {
            return false;
        }
        buffer.clear();
        connected = true;
        return true;
    }

    void close()
    {
        if (!connected) return;
        sock.closeConnection();
        connected = false;
        buffer.clear();
    }

    bool send(const std::string& packet)
    {
        if (!connected) return false;
        std::string line = packet;
        line.push_back('\n');
        if (!sock.sendData(&line[0], (unsigned int)line.size()))
        {
            connected = false;
            return false;
        }
        return true;
    }

    // Returns false when the connection dropped. `line` is empty when nothing was pending.
    bool poll(std::string& line)
    {
        line.clear();

        size_t newline = buffer.find('\n');
        if (newline == std::string::npos)
        {
            // select() with a zero timeout rather than a non-blocking recv: the socket carries the
            // 5s SO_RCVTIMEO ServerSocket sets for request/response use, and a blocking read here
            // would stall the coordinator for that long between draining worker hits.
            fd_set readable;
            FD_ZERO(&readable);
            FD_SET(sock.serverSocket, &readable);
            timeval immediately;
            immediately.tv_sec = 0;
            immediately.tv_usec = 0;

            const int ready = select((int)sock.serverSocket + 1, &readable, nullptr, nullptr, &immediately);
            if (ready < 0)
            {
                connected = false;
                return false;
            }
            if (ready == 0)
            {
                return true;
            }

            char chunk[4096];
            const int received = (int)recv(sock.serverSocket, chunk, sizeof(chunk), 0);
            if (received <= 0)
            {
                connected = false;
                return false;
            }
            buffer.append(chunk, received);
            newline = buffer.find('\n');
            if (newline == std::string::npos) return true;
        }

        line = buffer.substr(0, newline);
        buffer.erase(0, newline + 1);
        return true;
    }

    bool subscribe(const std::string& wallet, const std::string& worker)
    {
        char packet[512];
        snprintf(packet, sizeof(packet),
            "{\"id\":%d,\"wallet\":\"%s\",\"worker\":\"%s\",\"protocolVersion\":%d}",
            EVENT_SUBSCRIBE, wallet.c_str(), worker.c_str(), PROTOCOL_VERSION);
        return send(packet);
    }

    bool submit(const std::string& jobId, const std::string& computorId, const std::string& seed,
        const unsigned char* nonce, unsigned int claimedScore)
    {
        char packet[768];
        snprintf(packet, sizeof(packet),
            "{\"id\":%d,\"nonce\":\"%s\",\"seed\":\"%s\",\"computorId\":\"%s\","
            "\"jobId\":\"%s\",\"claimedScore\":%u}",
            EVENT_SUBMIT, toHex(nonce, 32).c_str(), seed.c_str(), computorId.c_str(),
            jobId.c_str(), claimedScore);
        return send(packet);
    }

    bool reportHashrate(const std::string& computorId, unsigned long long hashrate)
    {
        char packet[256];
        snprintf(packet, sizeof(packet),
            "{\"id\":%d,\"computorId\":\"%s\",\"hashrate\":%llu}",
            EVENT_REPORT_HASHRATE, computorId.c_str(), hashrate);
        return send(packet);
    }
};

}

# Qubic Reference Miner
This Repo contains the reference implementation of the algoritm used in Qubic.

## Licensing

This project is licensed under the **Anti-Military License**—see the `LICENSE` file for details.

### Third-Party Licenses

This project incorporates code from third-party sources which are governed by different licenses. Full compliance information, including the original copyright notices and terms for these dependencies, can be found in the **`NOTICE`** file in the repository root.

## File structure
- score_common.h: shared functions used for scoring (the random2 pool generator).
- Qiner.cpp: Contains the main process logic/functionality. Mainly shows how to communicate with the node and drive the miner.
- K12AndKeyUtill.h, keyUtils.h, keyUtils.cpp: Provide K12 and key conversion utilities/functions.

The algorithm-specific files (the scorer and its task format) are listed in the algorithm section.

# Requirement
- CPU: support at least AVX2 instruction set
- OS: Windows, Linux

# Build
## Windows
### Visual Studio 2022
- Open Qiner.sln
- Build
### Other Visual Studio versions

- Support generation using CMake with below command
```
# Assume in Qiner folder
mkdir build
cd build
"C:\Program Files\CMake\bin\cmake.exe" -G <Visual Studio Generator>
# Example: C:\Program Files\CMake\bin\cmake.exe" -G "Visual Studio 17 2022"
```
- Open Qiner.sln in build folder and build

### Enable AVX512
- Open Qiner.sln
- Right click Qiner->[C/C++]->[Code Generation]->[Enable Enhanced Instruction Set] -> [...AVX512] -> OK

## Linux
Currently support GCC and Clang
- Installed required libraries

For example,
- Ubuntu with GCC
```
sudo apt install build-essential
```
- Ubuntu with Clang
```
sudo apt install build-essential
sudo apt install clang
```


### GCC
Run below command
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Clang
Run below command
```
mkdir build
cd build
CC=clang CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Enable AVX512
To enable AVX512, -DENABLE_AVX512=1 need to be parse in the cmake command.

Example,
```
# GCC
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=1

# Clang
CC=clang CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=1
```

# Run
```
./Qiner <Node IP> <Node Port> <MiningID> <Signing Seed> <Mining Seed> <Number of threads>
```
The active algorithm may take an extra trailing argument - see its section below.

Example: 
```
./Qiner 192.168.1.2 31841 BZBQFLLBNCXEMGLOBHUVFTLUPLVCPQUASSILFABOFFBCADQSSUPNWLZBQEXK aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaa
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 8
```

# Algorithm 2026-07-16 (bpp9000)

## Files
- score_bpp9000.h: the bpp9000 scorer (a recurrent ternary-LUT network).
- task_file.h: the unified task-file format (topology + data blocks) with pack/parse and hash helpers.

## Run argument
The miner takes one optional trailing CLI argument - `[Task file path]` - the path to the bpp9000 task file; it defaults to `task_bpp9000.bin`.

## Overview
A recurrent ternary-LUT network scored over a windowed sequence. Given a task (a fixed network wiring plus a sequence of input/output samples), mining searches the per-neuron lookup tables (LUTs) for the configuration that reproduces the samples with the fewest errors. The score is that error count (lower is better). The task is loaded from a unified task file (defaults to `task_bpp9000.bin`).

## Key concepts
- Every value is a **trit** in `{0, 1, 2}`, where `2 = UNKNOWN`.
- The network has `P` neurons, each typed **input** / **output** / **evolution**; each neuron owns a `27`-entry **LUT** (`3^3`).
- A neuron's next value is `LUT[t0 + 3*t1 + 9*t2]`, indexed by its three neighbours' trits.
- Wiring is explicit per neuron (`neighborIndices[n*K + k]`, any neuron in `[0, P)`). One **signal neuron** self-clocks the input feeding.
- Only the LUTs change under mutation; the wiring and the sample data are fixed by the task.

## Random sources
Where every part of the network comes from. `random2` never invents bytes - it reads them out of the pool, and the "derived from" value only selects the read positions.

| Source | Derived from | When |
|---|---|---|
| `random2` pool | epoch spectrum digest | epoch start |
| Neuron placement (input / output / signal) | read from the task file | epoch start |
| Wiring (each neuron's 3 neighbours) | read from the task file | epoch start |
| Initial LUT contents | `K12(publicKey)` | per public key |
| Mutation seeds | `K12(publicKey \|\| nonce[3..31])` | per nonce |
| Algorithm select | `(nonce[0] & 1) != 0` -> bpp9000 (odd nonce; the miner mines odd nonces) | per nonce |
| L (LUT entries changed per mutation) | `nonce[1]` | per nonce |
| K (anti-attractor length) | `nonce[2]` | per nonce |

The random pool is built from the epoch spectrum digest, in Qiner it is seen as miningSeed, the public key decide the starting LUT ,the public key and nonce decide *where each draw reads from it*. Neuron placement and wiring are no longer random: they are read from the task file.

## Constants
```
K = NUMBER_OF_NEIGHBORS        // 3, hardcoded (LUT index is base-3 over 3 trits)
N = NUMBER_OF_INPUT_NEURONS
P = POPULATION_THRESHOLD       // total neurons, power of 2
T = SEQUENCE_LENGTH            // samples in the task
W = WINDOW_WIDTH               // samples fed per window
numberOfWindows = T - W        // graded windows per score()
maxTicks = MAX_NUMBER_OF_TICKS // per-window inference budget; exceeding it fails the process
S = NUMBER_OF_MUTATIONS        // search steps
```

## Code flow (pseudocode)
```
initialize(miningSeed, taskFile):
    generateRandom2Pool(miningSeed) -> pool
    loadTaskData(taskFile):
        parse + hash-verify the topology block (input / output / signal indices + neighbour wiring)
        parse + hash-verify the data block (packed input / output trit samples)
        derive neuron roles (input / output / evolution)

// The miner tries random odd nonces until one solves:
findSolution(publicKey, nonce):
    return computeScore(publicKey, nonce) <= SOLUTION_THRESHOLD

computeScore(publicKey, nonce):             // anti-attractor local search
    L     = clamp(nonce[1], 1, MAX_L)       // LUT flips applied per step
    Kexpl = clamp(nonce[2], 0, S)           // length of the explore phase
    cur   = initializeANN(publicKey, nonce)
    best  = cur
    for s in 0 .. S-1:
        save previous LUTs
        for i in 0 .. L-1:                       // apply L LUT flips this step
            mutate(mutationSeed[s * MAX_L + i])  // each flips one LUT entry
        r = score()
        accept = (s < Kexpl) ? (r >= cur)   // explore: allow equal-or-worse
                             : (r <= cur)   // exploit: keep equal-or-better
        if accept: cur = r  else: rollback to previous LUTs
        if cur < best: best = cur
    return best

initializeANN(publicKey, nonce):
    initial LUT    = random2(K12(publicKey), pool)                 // per computor, fixed
    mutation seeds = random2(K12(publicKey || nonce[3..31]), pool) // the search path (nonce[0..2] excluded)
    set neuron types + values (UNKNOWN); set current LUTs from the initial LUT
    return score()

score():                                    // windowed, self-clocked; lower is better
    failures = 0
    for window in 0 .. numberOfWindows-1:
        reset all neurons to UNKNOWN
        feed W input samples, paced by the signal neuron, then read the settled output
        if it does not settle within maxTicks: return INFINITE_ERROR   // fails the whole network
        if predicted output != expected output: failures++
    return failures

processTick():                              // one inference step
    for each non-input neuron:
        next value = LUT[t0 + 3*t1 + 9*t2] from its three neighbours
    commit the new value into every non-input neuron

mutate(seed):                               // one LUT flip
    pick one LUT entry of one non-input neuron; set it to a different trit
```

## Task file
Three parts: `[ header ][ topology block ][ data block ]`.

- **Header** - dimensions (`N, M, T, P, K`) plus a hash of each block; lets the miner confirm it loaded the intended task.
- **Topology block** - the fixed ANN wiring: which neurons are input / output / signal, and each neuron's neighbours. Defines the network structure and never changes during mining.
- **Data block** - the sample sequence the ANN is scored against: the input/output rows it must predict across each window.

Both blocks are KangarooTwelve-hashed against the header and rejected on mismatch. Byte-level layout is in `task_file.h`.

# Previous algorithms (history)

The algorithms below are retired - their code has been removed from the miner. They are kept here for historical reference.

# Algorithm 2025-05-15 (hyperidentity)

## Definitions and precondition
- The `random2` generator will be used consistently across the entire pipeline.
- Each neuron can hold a value of `-1`, `0`, or `1`.
- Synapse weights range within the continuous interval \([-1, 1]\).
- Every neuron has exactly `2M` **outgoing** synapses. Synapses with zero weight represent *no connection*.
- A synapse is considered to be **owned** by the neuron from which it originates.
- A **mining seed** is used to initialize the random values of both input and output neurons.
- The **nonce** and **public key** determine:
  - The random placement of input and output neurons on a ring,
  - The weights of synapses,
  - The method for selecting synapses during the **evolution** step.
- Symbols,
  - S: evolution step
  - P: max neurons population
  - R: the number of mismatch between expected output and computed ouput
## I. ANN Structure Initialization

Given `nonce` and `pubkey` as seeds, and constants `K`, `L`, `N`, `2M`:

1. Initialize `K + L` neurons arranged in a ring structure.
   - `K` input neurons and `L` output neurons are placed at random positions on the ring.
2. Initialize input and output neuron values randomly.
3. Initialize weights of `2M` synapses with random values in the range `[-1, 1]` (i.e., `-1`, `0`, or `1`).
4. Convert neuron values to **trits**:
   - Keep `1` as is.
   - Change `0` to `-1`.
   - This step occurs **only once**.
5. Run initial tick simulation to initialize the `R` value.

---

## II. Tick Simulation

1. For each neuron, compute the new value as: `new_value = sum(weight × connected_neuron_value)`
2. Clamp each neuron's value to the range `[-1, 1]`.
3. Stop the tick simulation if **any** of the following conditions are met:
- All output neurons have non-zero values.
- `N` ticks have passed.
- No neuron values change.

---

## III. Evolution and Simulation

1. Compute the initial `R_best` — the number of non-matching output bits.
2. Repeat the following mutation steps up to `S` times:
    - Randomly pick a synapse and change its weight:
      - Increase or decrease it by `1` (i.e., ±1).
      - If the new weight is within `[-1, 1]`, proceed.
      - If the new weight becomes `-2` or `2`:
        - Revert the weight to its original value.
        - Insert a **new neuron** immediately after the connected neuron.
        - The new neuron:
          - Copies all **incoming** synapses from the original neuron.
          - Copies only the **mutated** outgoing synapse; all others are set to `0`.
        - Remove any synapses exceeding the `2M` limit per neuron.
3. Remove any neurons (except input/output) that:
    - Have all zero **incoming** synapses, or
    - Have all zero **outgoing** synapses.
4. Stop the evolution if the number of neurons reaches the population limit `P`.
5. Run **Tick Simulation** again.
6. Compute the new `R` value:
    - If `R > R_best`, discard the mutation.
    - If `R ≤ R_best`, accept the mutation and update `R_best = R`.

# Algorithm 2025-12-10 (addition)

## Overview
Focused on implementing an Addition function. The core changes involve the training data set size, input/output representation, and the scoring mechanism.

## Key Changes from Original Algorithm

| Aspect | Original | New |
| -----  | -------- |-----|
| Input neurons | Random, value can be changed in tick simulation | Load from training data, value unchanged in tick simulation|
| Tick simulation | Run once per inference | Run 2^Input pairs|
| Score | Matching bits for 1 pattern | Total matching bits across ALL training pairs|
| Neighbor count | Fixed (always maxNeighbors) | Dynamic (min(maxNeighbors, population-1))|

## Pseudo code
```
// ========== CONSTANTS ==========
// Can be adjusted
K = NUMBER_OF_INPUT_NEURONS      // 14 (7 bits for A + 7 bits for B)
L = NUMBER_OF_OUTPUT_NEURONS     // 8 (8 bits for result C)
N = NUMBER_OF_TICKS              // 120
M = MAX_NEIGHBOR_NEURONS / 2     // 364 (half of 728)
S = NUMBER_OF_MUTATIONS          // 100
P = POPULATION_THRESHOLD         // K + L + S = 122

TRAINING_SET_SIZE = 2^K          // 16,384
MAX_SCORE = TRAINING_SET_SIZE × L  // 131,072
SOLUTION_THRESHOLD = MAX_SCORE × 4/5  // 104,857

// ========== I. NEW DATA STRUCTURES ==========
STRUCT Pair:
    char input[K]                // K/2 bits of A, K/2 bits of B (values: -1 or +1)
    char output[L]               // L bits of C (values: -1 or +1)

Pair allPairs[ALL_PAIRS_SIZE]    // All possible (A, B, C) combinations
Pair selected[SELECTED_SIZE]     // Randomly selected training pairs

// ========== II. INITIALIZATION ==========
FUNCTION initialize(publicKey, nonce):
    // 1. Generate random2 pool
    hash = KangarooTwelve(publicKey || nonce)
    initValue = Random2(hash)

    // 2. Generate all 2^K possible (A, B, C) pairs
    boundValue = 2^(K/2) / 2     // 64 for 7-bit signed [-64, 63]
    index = 0
    FOR A = -boundValue TO boundValue-1:
        FOR B = -boundValue TO boundValue-1:
            C = A + B            // C in range [-128, 126]
            allPairs[index].input[0..K/2-1] = toTernaryBits(A, K/2)   // 7 bits
            allPairs[index].input[K/2..K-1] = toTernaryBits(B, K/2)   // 7 bits
            allPairs[index].output = toTernaryBits(C, L)              // 8 bits
            index++
    
    // 3. Initialize ANN structure
    population = K + L
    // Randomize location of input neurons and output neurons
    randomizeNeuronTypes(initValue)  // K inputs, L outputs
    // Random weights of synapses
    initializeSynapseWeights(initValue)

    // 4. Fist inference for init best score
    inferANN()

// ========== III. SCORING ==========
FUNCTION inferANN():
    totalScore = 0

    // Evaluate ANN on all 2^K pairs
    FOR i = 0 TO ALL_PAIRS-1:
        // Load input values (these stay CONSTANT during ticks)
        setInputNeurons(selected[i].input)

        // Reset output neurons to 0
        resetOutputNeurons()

        // Run tick simulation
        runTickSimulation()

        // Count matching output bits
        FOR j = 0 TO L-1:
            IF outputNeuron[j].value == selected[i].output[j]:
                totalScore++

    RETURN totalScore

// ========== IV. TICK SIMULATION ==========
// Same as before, but:
// - Runs on K input neurons and L output neurons
// - Input neuron values are PRESERVED (not updated during ticks)
// - Uses dynamic neighbor count: min(MAX_NEIGHBOR_NEURONS, population - 1)

FUNCTION runTickSimulation():
    FOR tick = 0 TO N-1:
        // Calculate weighted sums for all neurons
        actualNeighbors = min(MAX_NEIGHBOR_NEURONS, population - 1)
        FOR each neuron n in population:
            sum = 0
            FOR each neighbor m within actualNeighbors:
                sum += neurons[m].value × synapses[n→m].weight
            neuronValueBuffer[n] = sum

        // Update only NON-INPUT neurons
        FOR each neuron n in population:
            IF neurons[n].type != INPUT:
                neurons[n].value = clamp(neuronValueBuffer[n], -1, +1)

        // Early exit conditions
        IF allNeuronsUnchanged() OR allOutputsNonZero():
            BREAK

// ========== V. MUTATION ==========
// Same as before
FUNCTION mutate(step):
    actualNeighbors = min(MAX_NEIGHBOR_NEURONS, population - 1)
    synapseIdx = random(initValue.synapseMutation[step]) % (population × actualNeighbors)

    IF currentWeight + mutation is valid (-1, 0, +1):
        synapse[synapseIdx].weight += mutation
    ELSE:
        // Weight overflow → INSERT new neuron
        insertNeuron(synapseIdx)
        population++

    // Remove redundant neurons (all-zero incoming OR outgoing synapses)
    WHILE hasRedundantNeurons():
        removeRedundantNeurons()

// ========== MAIN LOOP ==========
FUNCTION computeScore(publicKey, nonce):
    bestScore = initialize(publicKey, nonce)
    bestANN = copy(currentANN)

    FOR s = 0 TO S-1:
        mutate(s)

        IF population >= P:
            BREAK

        newScore = inferANN()

        IF newScore > bestScore:
            bestScore = newScore
            bestANN = copy(currentANN)
        ELSE:
            currentANN = copy(bestANN)  // Rollback

    RETURN bestScore
```

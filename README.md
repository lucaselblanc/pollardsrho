# Pollard's Rho Algorithm for SECP256K1 Curve (Beta)

![C++](https://img.shields.io/badge/Language-C++-blue)
![CUDA](https://img.shields.io/badge/Language-CUDA-green)
![CUDA](https://img.shields.io/badge/Arch-GPU%20&%20CPU-orange)
![Linux](https://img.shields.io/badge/Platform-Linux-white)

## Description

 This repository contains an implementation of Pollard’s Rho algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) on the secp256k1 curve. The objective is to recover the scalar k from the relation H = k * G, where G is the curve generator and H is a public point.

 The algorithm executes high-speed pseudo-random walks over the secp256k1 group using an R-adding walk iteration function. It utilizes a table of 2048 pre-computed steps and a MurmurHash3-based avalanche function to determine state transitions, maintaining the algebraic representation `R = a * G + b * H`. The scalars are initially probabilistically distributed within a specific ```key_range```, optimizing collision search for an initial probability distribution in O(√K) instead of the general distribution for the entire group O(√N), and as points are multiplied on the curve, walkers may eventually exceed ```key_range```, since the search in Pollard's rho algorithm is blind, and for the algorithm it does not matter in which scalar area the points actually are, as long as the walkers maintain their linear congruence relations for the group N.

 When two independent walkers encounter the same group element (a collision) with distinct coefficient pairs `(a, b)`, it yields a linear congruence modulo the group order `N`, allowing for the immediate recovery of the private key with the calculation of d through mod inversion. To maximize throughput and enable massive parallelization, the implementation employs a Distinguished Points (DP) strategy, where only points meeting a specific bit-mask criteria are stored in a high-concurrency hash map. This allows multiple CPU threads to traverse different paths simultaneously with minimal memory overhead. The system is specifically tuned for the secp256k1 curve and requires a Bitcoin public key as the target.

## Distinguished Points (DP)

 The Distinguished Points strategy is a memory-saving filter. Instead of storing every step of the walk (which would crash your RAM), the algorithm only saves points that satisfy a specific condition: the first d bits of the X-coordinate must be zero. When two walkers hit the same DP, a collision is found and the private key is recovered. ​The Trade-off: More DP bits = Less RAM used, but slower collision detection. Fewer DP bits = Faster detection, but higher RAM consumption.

#### Delay Of Distinguished Points

 When a walker begins traversing a path already explored by another walker, a collision will be delayed if the distinguished points filter condition is not met for both walkers. The delay will be overcome after the distinct points are recorded in the dp table. The higher the dp value, the greater the delay for a collision to be detected and recorded by the hashmap. To mitigate this, it would be necessary to disable the dp filter, but this would cause excessive RAM usage and would not be worth the effort. This delay is a necessary evil when using distinct points.

Theoretical Calculus:

```
int dp = (key_range / 2.0) - math.log2(RAM_BYTES / 128);
```

Simple Abstraction:

```
int dp = math.floor(key_range / 2.0);
```

```
k2 = 1
k4 = 2
k8 = 4
k16 = 8
k32 = 16
k64 = 32
k128 = 64
K256 = 128
```

## Pre-Computed Points ```windowSize``` in L2/L3 Caches

 For small ranges where collisions occur quickly, ```windowSize``` is calculated to have a larger table points that can occupy L3, since it is not necessary to extract the best performance for a collision that occurs in a few steps, with the use of a larger table, there are more points, reducing the chance of walkers entering short loops, because the entropy is greater. As the range increases, the walkers will have more space to explore, and it is at this point that the use of lower L2 latency is necessary. From >= 40 bits, ```windowSize``` starts to fit in L2, slightly overflowing into L3, which allows the ops/s speed to increase by ~50%, with less entropy of points in a much larger probability space, the path correlation of the walkers increases, and the state space of the transitions decreases, favoring the birthday paradox, as the trajectory of the walkers becomes more predictable.

## Algorithm Complexity

 The expected time complexity of Pollard's Rho algorithm for elliptic curves is O(√n), where n is the order of the group, in this implementation, the probability distribution in the initial steps is restricted to √k, subsequently encompassing the entire group in √n. Given secp256k1, this translates to approximately O(2^√N), as predicted by the birthday paradox for random walks over a finite group.

## Prerequisites

- g++
- boost/multiprecision/cpp_int.hpp

## Installation

1. Clone this repository:
    ```bash
    ~/$ git clone https://github.com/lucaselblanc/pollardsrho.git
    ```

2. Install the necessary libraries:
    ```bash
    sudo apt-get update
    sudo apt install build-essential g++ -y
    sudo apt install libboost-dev -y
    ```

3. Compile the project:
    ```bash
    ~/$cd pollardsrho
    ~/pollardsrho$ make
    ```

4. Run the program:
    ```bash
    ~/pollardsrho$ ./pollardsrho <compressed public key(hex)> <key range(int)> <OPTIONAL DP(int)>
    ```

    Replace `<compressed public key>` with the point \(G\) on the secp256k1 curve multiplied by your private key value, and `<key range>` with the size of the search interval for \(k\).

    Example usage:
    ```bash
    ~/pollardsrho$ ./pollardsrho 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 135
    ```

## Commands

 The random walk begins using the public point of the compressed public key as the parameter H, the target private key range for initializing the initial probability space, and the optional distinguished points parameter, which will be calculated automatically if not defined:
```bash
~/pollardsrho$ ./pollardsrho <compressed public key> <key range> <dp bits>
```

#### Negation Map:
 It is possible to enable point negation activating the flag **NEGATION_MAP_TRUE** to increase efficiency in ~1.41x √n, however there is a risk of short 2-step loops. The implementation detects short and long loops automatically through Brent's algorithm, although this is rarer for Pollard's rho algorithm and more frequent in Kangaroo Lambda:
```bash
~/pollardsrho$ ./pollardsrho 036ea839d22847ee1dce3bfc5b11f6cf785b0682db58c35b63d1342eb221c3490c 24 8 NEGATION_MAP_TRUE 
```

## Differences between Pollard's Rho and Pollard's Kangaroo/Lambda

#### Pollard's Rho Algorithm (Probabilistic √N)

 In the Pollard's Rho algorithm, it is assumed that we do not know the minimum-maximum scalar range in which the private key may lie, as the search is blind, this is useful when you have no prior knowledge of k, so the value of ```key_range``` can be any value >= 1. If you correctly guess the maximum range, you will eventually arrive at a solution, in the worst-case scenario in up to √K steps, following the birthday paradox on which the algorithm is based. The advantage of Pollard's Rho over Pollard's Kangaroo is the generality of the points, as the recovery of k does not depend on the distance between the two colliding walkers, but rather on the linear congruence relationship maintained between them. This algorithm rarely has problems with short or long loops due to its chaotic and completely pseudo-random walk, as there is no deterministic linearity dependency for a solution, The loops(fruitless cycles) occur due to this linearity combined with entropy density. The difference between the coefficients a and b is resolved through modular inversion and not with simple addition or subtraction as happens in Pollard's Kangaroo. Finally, any collision is valid for Pollard's Rho, which ends up being a huge advantage, due to the astronomically large group space of the secp256k1 curve, the program will never stop until a collision occurs, or it reaches the 256-bit limit, however, it is more likely that the operating system itself will terminate the process before reaching this limit, if the total number of operations exceeds √K, the probability of a useful collision occurring after that point is very low, but this is left to the solver's discretion.

#### Pollard's Kangaroo/Lambda Algorithm (Probabilistic √K)
 
 In Pollard's Kangaroo (lambda) algorithm, it is assumed that there is prior knowledge about k, the search is strictly limited to the ```key_range``` and will not go beyond it. Without the exact search range, the algorithm becomes impracticable, because the path taken by the domesticated kangaroo and the wild kangaroo is deterministic, and it is fully calculated using the average steps of √K, without the original range, the algorithm breaks. It's as if Pollard's Kangaroo were a more efficient version of the Baby Steps Giant Steps (BSGS) algorithm, the difference is that lambda does this in a more intelligent (no brute force) way, using the domesticated kangaroo to initiate the search by scattering traps/points along the path, which are saved in a table. Eventually, a wild kangaroo has a statistical chance of landing on one of these points left by the tame, causing a collision and solving k. Due to this determinism of the path, the recovery of k occurs through a simple addition or subtraction of the accumulated distance between tame and wild, however, this simplicity comes at a high price with the increase in internal loops where the kangaroo gets stuck on a path it has already visited, another bad scenario occurs when a wild kangaroo makes a very large jump and ends up never landing on any of the points left by the domesticated kangaroo, meaning the wild kangaroo never has a chance to resolve any future collisions, in this algorithm, the main concern will always be to get a kangaroo out of a loop when it gets stuck in one, and to normalize the average of the jumps to avoid overshooting. Both the rho and the kangaroo algorithms use distinguished points to reduce memory costs when saving points.

## External Libraries Used

<cuda.h>
<cuda_runtime.h>
"secp256k1.h"
"parallel_hashmap/phmap.h"

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Add a Star: <a href="https://github.com/lucaselblanc/pollardsrho/stargazers"><img src="https://img.shields.io/github/stars/lucaselblanc/pollardsrho?style=flat-square" alt="GitHub stars" style="vertical-align: bottom; width: 65px; height: auto;"></a>

## Donations: bc1pxqwuyfwvttjgttfmpt0gk0n7yzw3k7cyzzpc3rsc4lumr8ywythsj0rrhd

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

<p align="center">
  <a href="https://github.com/lucaselblanc">
    <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=18&duration=2000&pause=100&multiline=true&width=500&height=80&lines=Lucas+Leblanc;Programmer+%7C+Student+%7C+Cyber+Security;+%7C+Android+%7C+Apps" alt="Typing SVG" />
  </a>
</p>

<a href="https://github.com/lucaselblanc">
    <img src="https://github-stats-alpha.vercel.app/api?username=lucaselblanc&cc=22272e&tc=37BCF6&ic=fff&bc=0000">
</a>

- 🔭 I’m currently working on [Data Leak Search](https://play.google.com/store/apps/details?id=com.NoClipStudio.DataBaseSearch)

- 🚀 I’m looking to collaborate on: [Cyber-Security](https://play.google.com/store/apps/details?id=com.hashsuite.droid)

- 📝 I regularly read: [https://github.com/bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1)

- 📄 Know about my experiences: [https://www.linkedin.com/in/lucas-leblanc-215594208](https://www.linkedin.com/in/lucas-leblanc-215594208)

<br>
My Github Stats

![](http://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=lucaselblanc&theme=dracula) 
![](http://github-profile-summary-cards.vercel.app/api/cards/repos-per-language?username=lucaselblanc&theme=dracula) 
![](http://github-profile-summary-cards.vercel.app/api/cards/most-commit-language?username=lucaselblanc&theme=dracula)

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://www.linkedin.com/in/lucas-leblanc-215594208" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="lucas-leblanc-215594208" height="30" width="40" /></a>
<a href="https://www.youtube.com/@noclipstudiobr" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/youtube.svg" alt="@noclipstudiobr" height="30" width="40" /></a>
<a href="https://discord.gg/https://discord.gg/wXqcJDHht8" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/discord.svg" alt="https://discord.gg/wXqcJDHht8" height="30" width="40" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://developer.android.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/android/android-original-wordmark.svg" alt="android" width="40" height="40"/> </a> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/c/c-original.svg" alt="c" width="40" height="40"/> </a> <a href="https://www.w3schools.com/cpp/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg" alt="cplusplus" width="40" height="40"/> </a> <a href="https://www.w3schools.com/cs/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/csharp/csharp-original.svg" alt="csharp" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/firebase/firebase-original.svg" alt="firebase" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://unity.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/unity3d/unity3d-icon.svg" alt="unity" width="40" height="40"/> </a> </p>

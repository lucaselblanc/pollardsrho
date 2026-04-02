# Pollard's Rho Lambda Algorithm for SECP256K1 Curve (ρλ)

![C++](https://img.shields.io/badge/Language-C++-blue)
![CUDA](https://img.shields.io/badge/Language-CUDA-green)
![CUDA](https://img.shields.io/badge/Arch-GPU%20&%20CPU-orange)
![Linux](https://img.shields.io/badge/Platform-Linux-white)

## Fast Links:

[Main Index](https://github.com/lucaselblanc/pollardsrho/tree/main?tab=readme-ov-file#pollards-rho-lambda-algorithm-for-secp256k1-curve-%CF%81%CE%BB)

[Description](https://github.com/lucaselblanc/pollardsrho/tree/main?tab=readme-ov-file#description)

[Algorithm Version](https://github.com/lucaselblanc/pollardsrho/tree/main#pollards-rho-lambda-%CF%81%CE%BB)

[Benchmark](https://github.com/lucaselblanc/pollardsrho/tree/main#benchmark-tpu-v5e-8)

[Technical Features](https://github.com/lucaselblanc/pollardsrho/tree/main#technical-features)

[Distinguished Points](https://github.com/lucaselblanc/pollardsrho/tree/main#distinguished-points-dp)

[Delay Of Distinguished Points](https://github.com/lucaselblanc/pollardsrho/tree/main#delay-of-distinguished-points)

[Algorithm Complexity](https://github.com/lucaselblanc/pollardsrho/tree/main#algorithm-complexity)

[Prerequisites](https://github.com/lucaselblanc/pollardsrho/tree/main#prerequisites)

[Installation](https://github.com/lucaselblanc/pollardsrho/tree/main#installation)

[Commands](https://github.com/lucaselblanc/pollardsrho/tree/main#commands)

[External Libraries Used](https://github.com/lucaselblanc/pollardsrho/tree/main#external-libraries-used)

## Description

 This repository contains a high-performance implementation of Pollard’s Rho Lambda algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) on the secp256k1 curve.

#### Pollard's Rho Lambda (ρλ)

 The algorithm executes high-speed pseudo-random walks over the secp256k1 group using an R-adding walk iteration function. It utilizes a table of 2048 pre-computed steps and a MurmurHash3-based avalanche function to determine state transitions, maintaining the algebraic representation `R = a * G + b * H`. The scalars are probabilistically distributed within a specific ```key_range```, optimizing collision search for an probability distribution in O(√K).

 When two independent walkers encounter the same group element (a collision) with distinct coefficient pairs `(a, b)`, it yields a linear congruence modulo the group order `N`, allowing for the immediate recovery of the private key with the calculation of d through mod inversion. To maximize throughput and enable massive parallelization, the implementation employs a Distinguished Points (DP) strategy, where only points meeting a specific bit-mask criteria are stored in a high-concurrency hash map. This allows multiple CPU threads to traverse different paths simultaneously with minimal memory overhead. The system is specifically tuned for the secp256k1 curve and requires a Bitcoin public key as the target.

## Benchmark TPU v5e-8

```
5 bits ≈ 00:00:00
10 bits ≈ 00:00:00
15 bits ≈ 00:00:00
20 bits ≈ 00:00:00
25 bits ≈ 00:00:00
30 bits ≈ 00:00:01
35 bits ≈ 00:00:01
40 bits ≈ 00:00:01
45 bits ≈ 00:00:06
50 bits ≈ 00:00:27
55 bits ≈ 00:02:25
60 bits ≈ 00:10:22
```

## Technical Features

#### Batch Jacobian-to-Affine (Montgomery Trick)

 Inversions in finite fields are computationally expensive. Both versions utilize Batch Inversion, processing multiple walkers simultaneously. This allows the algorithm to perform only one modular inversion per batch, converting Jacobian coordinates to Affine at a fraction of the usual cost.

#### Pre-Computed Points ```windowSize``` in L2/L3 Caches

 For small ranges where collisions occur quickly, ```windowSize``` is calculated to have a larger table points that can occupy L3, since it is not necessary to extract the best performance for a collision that occurs in a few steps, with the use of a larger table, there are more points, reducing the chance of walkers entering short loops, because the entropy is greater. As the range increases, the walkers will have more space to explore, and it is at this point that the use of lower L2 latency is necessary. If expected steps > lSize, ```windowSize``` starts to fit in L2, slightly overflowing into L3, which allows the ops/s speed to increase by ~50%, with less entropy of points in a much larger probability space, the path correlation of the walkers increases, and the state space of the transitions decreases, favoring the birthday paradox, as the trajectory of the walkers becomes more predictable.

#### Distinguished Points (DP)

 The Distinguished Points strategy is a memory-saving filter. Instead of storing every step of the walk (which would crash your RAM), the algorithm only saves points that satisfy a specific condition: the first d bits of the X-coordinate must be zero. When two walkers hit the same DP, a collision is found and the private key is recovered. ​The Trade-off: More DP bits = Less RAM used, but slower collision detection. Fewer DP bits = Faster detection, but higher RAM consumption.

#### Delay Of Distinguished Points

 When a walker begins traversing a path already explored by another walker, a collision will be delayed if the distinguished points filter condition is not met for both walkers. The delay will be overcome after the distinct points are recorded in the dp table. The higher the dp value, the greater the delay for a collision to be detected and recorded by the hashmap. To mitigate this, it would be necessary to disable the dp filter, but this would cause excessive RAM usage and would not be worth the effort, and would ruin the performance. This delay is a necessary evil when using distinct points.

Theoretical Calculus:

```
int dp = (key_range / 2.0) - math.log2(RAM_BYTES / POINT_BYTES);
```

Simple Abstraction:

```
int dp = math.sqrt(key_range);
```

```
k2 ≈ 1
k4 = 2
k8 ≈ 3
k16 ≈ 4
k32 ≈ 6
k64 = 8
k128 ≈ 11
K256 = 16
```

## Algorithm Complexity

 The expected time complexity of Pollard's Rho Lambda algorithm for elliptic curves is O(√n), where n is the order of the group, in this implementation, the probability distribution in the steps is restricted to O(√k) through an artificial cyclic subgroup, keeps the probabilistic window restricted to the range. Given secp256k1, this translates to approximately O(√range), as predicted by the birthday paradox for random walks over a finite group.

## Prerequisites

- g++
- boost/multiprecision/cpp_int.hpp
- libssl-dev

## Installation

1. Clone this repository:
    ```bash
    ~/$ git clone https://github.com/lucaselblanc/pollardsrho.git
    ```

2. Install the necessary libraries:
    ```bash
    sudo apt update
    sudo apt install build-essential g++ -y
    sudo apt install libboost-dev -y
    sudo apt install libssl-dev
    ```

3. Compile the project:
    ```bash
    ~/$cd pollardsrho
    ```

    ```bash
    ~/pollardsrho$ make
    ```

4. Run the program:
    ```bash
    ~/pollardsrho$ ./pollardsrho <compressed public key(hex)> <key range(int)> <walkers(int)> <OPTIONAL DP(int)>
    ```

    Replace `<compressed public key>` with the point \(G\) on the secp256k1 curve multiplied by your private key value, and `<key range>` with the size of the search interval for \(k\).

    Example usage:
    ```bash
    ~/pollardsrho$ ./pollardsrho 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 135 1000000 12
    ```

## Commands

 The random walk begins using the public point of the compressed public key as the parameter H, the target private key range for initializing the initial probability space, and the optional distinguished points parameter, which will be calculated automatically if not defined:
```bash
~/pollardsrho$ ./pollardsrho <compressed public key> <key range> <walkers> <dp bits>
```

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
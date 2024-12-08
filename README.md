# Pollard's Rho Algorithm for SECP256K1 Curve (Beta)

![C++](https://img.shields.io/badge/language-C++-blue)

## Description

This repository contains the implementation of Pollard's Rho algorithm for the secp256k1 elliptic curve. The goal is to generate private keys over O√n using a tortoise and many hares, similar to Pollard's Kangoroo, to run the program it is necessary to have the public key of the bitcoin wallet.

## Algorithm Complexity

The expected time complexity of Pollard's Rho algorithm for elliptic curves is <code>O(&#8730;n)</code>, where <code>n</code> is the order of the group to the generating point(G). Given secp256k1, this translates to approximately <code>O(2<sup>128</sup>)</code>.

#### Prerequisites

- boost::multiprecision.

---

## Installation

1. Clone this repository:
    ```bash
    ~/$ git clone https://github.com/lucaselblanc/pollardsrho.git
    ```

2. Install the necessary libraries:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install gcc g++
    sudo apt-get install libboost-all-dev
    ```

3. Compile the project:
    ```bash
    ~/$cd pollardsrho
    ~/pollardsrho$ make
    ```

4. Run the program:
    ```bash
    ~/pollardsrho$ ./pollardsrho <compressed public key> <key range>
    ```

    Replace `<compressed public key>` with the point \(G\) on the secp256k1 curve multiplied by your private key value, and `<key range>` with the size of the search interval for \(k\).

    Example usage:
    ```bash
    ~/pollardsrho$ ./pollardsrho 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 135
    ```

## Commands

- `~/pollardsrho$ ./pollardsrho <compressed public key> <key range>`: Starts the search for the private key corresponding to the given public key.

## Libraries Used

- secp256k1.h
- random
- thread
- mutex
- atomic
- unordered_set
- omp.h
- boost/multiprecision/cpp_int.hpp

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Add a Star: <a href="https://github.com/lucaselblanc/pollardsrho/stargazers"><img src="https://img.shields.io/github/stars/lucaselblanc/pollardsrho?style=flat-square" alt="GitHub stars" style="vertical-align: bottom; width: 65px; height: auto;"></a>

## Donations: bc1pxqwuyfwvttjgttfmpt0gk0n7yzw3k7cyzzpc3rsc4lumr8ywythsj0rrhd

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

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

- 📄 Know about my experiences: [https://www.linkedin.com/in/lucas-jhonatan-215594208](https://www.linkedin.com/in/lucas-jhonatan-215594208)

<br>
My Github Stats

![](http://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=lucaselblanc&theme=dracula) 
![](http://github-profile-summary-cards.vercel.app/api/cards/repos-per-language?username=lucaselblanc&theme=dracula) 
![](http://github-profile-summary-cards.vercel.app/api/cards/most-commit-language?username=lucaselblanc&theme=dracula)

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/lucas-jhonatan-215594208" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="lucas-jhonatan-215594208" height="30" width="40" /></a>
<a href="https://www.youtube.com/@noclipstudiobr" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/youtube.svg" alt="@noclipstudiobr" height="30" width="40" /></a>
<a href="https://discord.gg/https://discord.gg/wXqcJDHht8" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/discord.svg" alt="https://discord.gg/wXqcJDHht8" height="30" width="40" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://developer.android.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/android/android-original-wordmark.svg" alt="android" width="40" height="40"/> </a> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/c/c-original.svg" alt="c" width="40" height="40"/> </a> <a href="https://www.w3schools.com/cpp/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg" alt="cplusplus" width="40" height="40"/> </a> <a href="https://www.w3schools.com/cs/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/csharp/csharp-original.svg" alt="csharp" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/firebase/firebase-original.svg" alt="firebase" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://unity.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/unity3d/unity3d-icon.svg" alt="unity" width="40" height="40"/> </a> </p>

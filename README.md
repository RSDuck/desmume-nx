# DeSmuME for Nintendo Switch

This is an unfinished port of the Nintendo DS emulator DeSmuME for Nintendo Switch. It was originally based of @masterfeizz's [port](https://github.com/nuxdie/DeSmuME-NX) which is probably based off the last release of DeSmuME from a few years ago. I later ported all changes so that it's now based off the [Git master branch of DeSmuME](https://github.com/TASVideos/desmume).

The emulator runs at near fullspeed without any overclocking (in 2D only games), though only overclocking to 1.78 GHz results in games running at fullspeed. In moments where new code is compiled via the JIT lag spikes occur and the generated code can still be optimised. The latter can probably be mitigated by optimising the JIT as detailed in the next section.

I worked on this project from somewhere around October 2018 until the beginning of December. The developement halted mostly because I'm not satisfied with the result and at the same time the melonDS archives similar levels of performance with full overclocking while being far more accurate.

Followup: I actually started working on this again. I optimised the JIT (a bit) and implemented the OGL renderer and it still runs bad. What an disappointment.

## JIT

The largest addition of mine is a newly written JIT backend for the ARMv8 architecture. It is the first JIT compiler I've ever written and it took several iterations to get it right. It uses the [code emitter of the Dolphin project](https://github.com/dolphin-emu/dolphin/tree/master/Source/Core/Common) (huge credits to them!).

The code generation can still be improved. The native NZCV status register is currently retrived after every instruction which sets flags via a `MRS` instruction which probably isn't meant to be used that often. Instead leaving them in the native register and only retrieving them when it's read or the register is used for internal purposes would be a possible solution.

Followup: I actually implemented what's detailed in the paragraph above. I also implemented that short JIT blocks which end with an unconditional branch will be called directly after each other.

You can find it and three older iterations in `desmume/src/utils/arm_arm64`.

## sse2neon

A slightly patched version of [sse2neon](https://github.com/jserv/sse2neon) is used to get atleast a bit of out of the large amount of handwritten SIMD code.

After trying some games without sse2neon some games seem to have broken graphics when run using the non vectorised code.

## OpenGL backend

The OpenGL backend is now ported and it's, eh underwhelming. To make it work I generated my own GLAD OpenGL binding which supports compability mode profiles and a ton of extensions which are incorparted into the standard years ago since DeSmuME supports stone age GPUs. It doesn't run much better than the software renderer.
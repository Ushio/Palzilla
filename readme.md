## Palzilla - Path Cuts Like Full-Spectrum Caustics Renderer for RTCamp 11

![image](teaser.png)

[RTCamp 11](https://sites.google.com/view/rtcamp11)

## Features
- Path Cuts based deterministic specular path calculation
- Fast Photon tracing based tuples
- No linear algebra external libs
- Pure software ray tracing (Simple LBVH)

## Build
```
git submodule update --init
premake5 vs2022
```

## Presentations
[Short Introduction](presentations/Palzilla.pdf)

[Seminar - こうろせつだん！](presentations/こうろせつだん.pdf)

## References

- きなこもち, Pathtracing technics for Caustics ( https://speakerdeck.com/kinakomoti321/pathtracing-technics-for-caustics-f93e343a-2d66-4d7c-a78f-23a20a7968af )
- きなこもち, Specular Polynomial ( https://scrapbox.io/ShaderMemorandom/Specular_Polynomial )
- @ykozw88, Interval Arithmetic, Affine Arithmetic ( https://speakerdeck.com/ykozw/ahuinyan-suan )
- @Shocker_0x15, アフィン演算で求める三角形上の関数の範囲 ( https://qiita.com/shocker-0x15/items/f2d7f6135c1bbfa16859 )
- Walter, et al. “Single Scattering in Refractive Media with Triangle Mesh Boundaries” [2009]
- Jakob and Marschner, ” Manifold exploration: a Markov Chain Monte Carlo technique for rendering scenes with difficult specular transport” [2012]
- Hanika, et al., ”Manifold Next Event Estimation” [2015]
- Hanika et al., “Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints” [2020]
- Wang, at al, “Path Cuts: Efficient Rendering of Pure Specular Light Transport” [2020]
- Fan, at al, ”Specular Polynomials” [2024]
    - https://github.com/mollnn/spoly
- Fan, at al, “Bernstein Bounds for Caustics” [2025]

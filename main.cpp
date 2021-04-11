#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <vector>
#include <random>
#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


static const float c_pi = 3.14159265359f;
static const float c_twoPi = 2.0f * c_pi;

typedef std::array<int, 2> IVec2;
typedef std::array<float, 2> Vec2;

template <typename T>
T Min(T a, T b)
{
    return a <= b ? a : b;
}

template <typename T>
T Max(T a, T b)
{
    return a >= b ? a : b;
}

template <typename T, size_t N>
std::array<T, N> operator -(const std::array<T, N>& A, const std::array<T, N>& B)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] - B[i];
    return ret;
}

template <typename T, size_t N>
std::array<T, N> operator +(const std::array<T, N>& A, const std::array<T, N>& B)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] + B[i];
    return ret;
}

template <typename T, size_t N>
std::array<T, N> operator *(const std::array<T, N>& A, const T& B)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] * B;
    return ret;
}

template <typename T, size_t N>
std::array<T, N> operator /(const std::array<T, N>& A, const T& B)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] / B;
    return ret;
}

template <typename T, size_t N>
T Dot(const std::array<T, N>& A, const std::array<T, N>& B)
{
    T ret = 0.0f;
    for (size_t i = 0; i < N; ++i)
        ret += A[i] * B[i];
    return ret;
}

template <typename T>
T Clamp(T value, T min, T max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

inline float SmoothStep(float edge0, float edge1, float x)
{
    if (edge0 == edge1)
        return edge0;

    x = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

inline float Lerp(float A, float B, float t)
{
    return A * (1.0f - t) + B * t;
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename LAMBDA>
float PerlinNoise(const IVec2& pixelPos, int cellSize, const LAMBDA& UnitVectorAtCell)
{
    auto RandomUnitVectorAtPos = [](const IVec2& pos)
    {
        size_t seed = 0x1337beef;
        hash_combine(seed, pos[0]);
        hash_combine(seed, pos[1]);

        std::mt19937 rng((unsigned int)seed);
        std::uniform_real_distribution<float> dist(0.0f, c_twoPi);

        float angle = dist(rng);

        return Vec2
        {
            cos(angle),
            sin(angle)
        };
    };

    auto DotGridGradient = [&](const IVec2& cell, const IVec2& pos)
    {
        Vec2 gradient = RandomUnitVectorAtPos(cell);
        IVec2 delta = pos - cell;
        Vec2 deltaf = Vec2{ (float)delta[0], (float)delta[1] } / float(cellSize);
        return Dot(deltaf, gradient);
    };

    IVec2 cellIndex = IVec2
    {
        (pixelPos[0] / cellSize),
        (pixelPos[1] / cellSize)
    };
    IVec2 cellFractionI = pixelPos - (cellIndex * cellSize);
    Vec2 cellFraction = Vec2
    {
        float(cellFractionI[0]) / float(cellSize),
        float(cellFractionI[1]) / float(cellSize),
    };

    // smoothstep the UVs
    cellFraction[0] = SmoothStep(0.0f, 1.0f, cellFraction[0]);
    cellFraction[1] = SmoothStep(0.0f, 1.0f, cellFraction[1]);

    // get the 4 corners of the square
    float dg_00 = DotGridGradient((cellIndex + IVec2{ 0,0 }) * cellSize, pixelPos);
    float dg_01 = DotGridGradient((cellIndex + IVec2{ 0,1 }) * cellSize, pixelPos);
    float dg_10 = DotGridGradient((cellIndex + IVec2{ 1,0 }) * cellSize, pixelPos);
    float dg_11 = DotGridGradient((cellIndex + IVec2{ 1,1 }) * cellSize, pixelPos);

    // X interpolate
    float dg_x0 = Lerp(dg_00, dg_10, cellFraction[0]);
    float dg_x1 = Lerp(dg_01, dg_11, cellFraction[0]);

    // Y interpolate
    return Lerp(dg_x0, dg_x1, cellFraction[1]);
}

template <typename LAMBDA>
float PerlinNoiseOctaves(const IVec2& pos, int octaves, int cellSize, const LAMBDA& UnitVectorAtCell)
{
    float ret = 0.0f;
    int frequency = 1;
    float amplitude = 1.0;

    for (int i = 0; i < octaves; ++i)
    {
        ret += PerlinNoise(pos * frequency, cellSize, UnitVectorAtCell) * amplitude;
        amplitude *= 0.5;
        frequency *= 2;
    }

    return ret;
}

template <typename LAMBDA>
void MakePerlinNoise(const char* fileName, int imageSize, int cellSize, int octaves, const LAMBDA& UnitVectorAtCell)
{
    std::vector<unsigned char> pixels(imageSize * imageSize);
    std::vector<float> pixelsf(imageSize * imageSize);

    float minValue = FLT_MAX;
    float maxValue = -FLT_MAX;

    float* pixelf = pixelsf.data();
    for (int iy = 0; iy < imageSize; ++iy)
    {
        for (int ix = 0; ix < imageSize; ++ix)
        {
            *pixelf = PerlinNoiseOctaves(IVec2{ ix, iy }, octaves, cellSize, UnitVectorAtCell);
            minValue = Min(minValue, *pixelf);
            maxValue = Max(maxValue, *pixelf);
            pixelf++;
        }
    }

    pixelf = pixelsf.data();
    unsigned char* pixelc = pixels.data();
    for (int iy = 0; iy < imageSize; ++iy)
    {
        for (int ix = 0; ix < imageSize; ++ix)
        {
            float value = Clamp((*pixelf - minValue) / (maxValue - minValue), 0.0f, 1.0f);
            *pixelc = (unsigned char)Clamp(value * 256.0f, 0.0f, 255.0f);
            pixelf++;
            pixelc++;
        }
    }

    stbi_write_png(fileName, imageSize, imageSize, 1, pixels.data(), 0);
}

int main(int argc, char** argv)
{
    static const int c_imageSize = 256;
    static const int c_cellSize = 16;
    static const int c_numCells = c_imageSize / c_cellSize;

    {
        std::vector<Vec2> vectors(c_numCells * c_numCells);
        std::mt19937 rng;
        std::uniform_real_distribution<float> dist(0.0f, c_twoPi);
        for (Vec2& v : vectors)
        {
            float angle = dist(rng);
            v = Vec2
            {
                cos(angle),
                sin(angle)
            };
        }

        auto UnitVectorAtCell = [&vectors](const IVec2& pos, int octave)
        {
            return vectors[pos[1] * c_numCells + pos[0]];
        };

        MakePerlinNoise("perlin_white_same_1.png", c_imageSize, c_cellSize, 1, UnitVectorAtCell);
        MakePerlinNoise("perlin_white_same_2.png", c_imageSize, c_cellSize, 2, UnitVectorAtCell);
        MakePerlinNoise("perlin_white_same_3.png", c_imageSize, c_cellSize, 3, UnitVectorAtCell);
    }
}

/*

TODO:
* make perlin noise with white noise
* make perlin noise with blue noise
* make perlin nosie with IGN for fun?
* DFT them all? with a python script!
* check notes in your email.
* could omp this to make it faster

? should multiple octaves use the same noise or different? turns out either is ok per eevee? maybe do and show both

? single ring images / DFTs vs multi ring.
? vary params like number of octaves and cell size. could make a grid of params i guess and label em with python?

Eevee's perlin noise tutorial: https://eev.ee/blog/2016/05/29/perlin-noise/
 * Also use the other method he talks about of selecting pre-made vectora.
 * Envies post talks about clumps being bad, so blue seems like a good idea.

*/
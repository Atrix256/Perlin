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

enum class SmoothingMode
{
    None,
    Smooth,
    Smoother
};

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

float Fract(float x)
{
    return x - floorf(x);
}

inline float SmoothStep(float edge0, float edge1, float x)
{
    if (edge0 == edge1)
        return edge0;

    x = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

inline float SmootherStep(float edge0, float edge1, float x)
{
    if (edge0 == edge1)
        return edge0;

    x = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);

    return 6.0f * x * x * x * x * x - 15.0f * x * x * x * x + 10.0f * x * x * x;
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

float Smooth(float f, SmoothingMode smoothing)
{
    switch (smoothing)
    {
        case SmoothingMode::Smooth:
        {
            return SmoothStep(0.0f, 1.0f, f);
        }
        case SmoothingMode::Smoother:
        {
            return SmootherStep(0.0f, 1.0f, f);
            break;
        }
        case SmoothingMode::None:
        default:
        {
            return f;
            break;
        }
    }
}

template <typename LAMBDA>
float PerlinNoise(const IVec2& pixelPos, int cellSize, int octave, SmoothingMode smoothing, const LAMBDA& UnitVectorAtCell)
{
    auto DotGridGradient = [&](const IVec2& cell, const IVec2& pos)
    {
        Vec2 gradient = UnitVectorAtCell(cell / cellSize, octave);
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
    cellFraction[0] = Smooth(cellFraction[0], smoothing);
    cellFraction[1] = Smooth(cellFraction[1], smoothing);

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
float PerlinNoiseOctaves(const IVec2& pos, int octaves, int cellSize, SmoothingMode smoothing, const LAMBDA& UnitVectorAtCell)
{
    float ret = 0.0f;
    int frequency = 1;
    float amplitude = 1.0;

    for (int i = 0; i < octaves; ++i)
    {
        ret += PerlinNoise(pos * frequency, cellSize, i, smoothing, UnitVectorAtCell) * amplitude;
        amplitude *= 0.5;
        frequency *= 2;
    }

    return ret;
}

template <typename LAMBDA>
void MakePerlinNoise(const char* fileName, int imageSize, int cellSize, int octaves, SmoothingMode smoothing, const LAMBDA& UnitVectorAtCell)
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
            *pixelf = PerlinNoiseOctaves(IVec2{ ix, iy }, octaves, cellSize, smoothing, UnitVectorAtCell);
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

Vec2 R2(int index)
{
    // Generalized golden ratio to 2d.
    // Solution to x^3 = x + 1
    // AKA plastic constant.
    // from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    float g = 1.32471795724474602596f;
    float a1 = 1.0f / g;
    float a2 = 1.0f / (g * g);

    float x = 0.5f;
    float y = 0.5f;
    for (int i = 0; i < index; ++i)
    {
        x = Fract(x + a1);
        y = Fract(y + a2);
    }
    return Vec2{ x, y };
}

// Interleaved Gradient Noise - a spatial low discrepancy sequence with good properties under TAA that does 3x3 neighborhood
// sampling for history rejection.
// Talked about in http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
// Presented at siggraph in 2014 in http://advances.realtimerendering.com/s2014/ part 1.
float IGN(int _x, int _y)
{
    float x = float(_x);
    float y = float(_y);
    return Fract(52.9829189f * Fract(0.06711056f * float(x) + 0.00583715f * float(y)));
}

int main(int argc, char** argv)
{
    static const int c_imageSize = 256;
    static const int c_cellSize = 16;
    static const int c_numCells = c_imageSize / c_cellSize;

    // A study in different sized cells
    {
        std::vector<Vec2> vectors(c_imageSize * c_imageSize);
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
            int x = pos[0];
            int y = pos[1];
            return vectors[y * c_imageSize + x];
        };

        MakePerlinNoise("perlin_2.png", c_imageSize, 2, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_4.png", c_imageSize, 4, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_8.png", c_imageSize, 8, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_16.png", c_imageSize, 16, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_32.png", c_imageSize, 32, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_64.png", c_imageSize, 64, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_64_smooth.png", c_imageSize, 64, 1, SmoothingMode::Smooth, UnitVectorAtCell);
        MakePerlinNoise("perlin_64_none.png", c_imageSize, 64, 1, SmoothingMode::None, UnitVectorAtCell);
        MakePerlinNoise("perlin_128.png", c_imageSize, 128, 1, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // white noise, with the same vectors for each octave
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
            int x = pos[0] % c_numCells;
            int y = pos[1] % c_numCells;
            return vectors[y * c_numCells + x];
        };

        MakePerlinNoise("perlin_white_same_1.png", c_imageSize, c_cellSize, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_white_same_2.png", c_imageSize, c_cellSize, 2, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_white_same_3.png", c_imageSize, c_cellSize, 3, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // white noise, with different vectors per octave
    {
        std::vector<Vec2> vectors(c_numCells * c_numCells * 3);
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
            int x = pos[0] % c_numCells;
            int y = pos[1] % c_numCells;

            int offset = 0;
            switch (octave)
            {
                case 0: offset = c_numCells * c_numCells * 0; break;
                case 1: offset = c_numCells * c_numCells * 1; break;
                case 2: offset = c_numCells * c_numCells * 2; break;
            }

            return vectors[y * c_numCells + x + offset];
        };

        MakePerlinNoise("perlin_white_different_1.png", c_imageSize, c_cellSize, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_white_different_2.png", c_imageSize, c_cellSize, 2, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_white_different_3.png", c_imageSize, c_cellSize, 3, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // blue noise, same vectors per octave
    {
        // Load the 16x16 blue noise
        std::vector<Vec2> vectors(c_numCells * c_numCells);
        {
            int w, h, c;
            uint8_t* pixels = stbi_load("BlueNoise16.png", &w, &h, &c, 4);

            if (w != c_numCells || h != c_numCells)
            {
                printf("ERROR! Invalid size blue noise texture!");
                return 1;
            }

            for (size_t index = 0; index < c_numCells * c_numCells; ++index)
            {
                float angle = c_twoPi * float(pixels[index * 4 + 0]) / 255.0f;
                vectors[index] = Vec2
                {
                    cos(angle),
                    sin(angle)
                };
            }

            stbi_image_free(pixels);
        }

        auto UnitVectorAtCell = [&vectors](const IVec2& pos, int octave)
        {
            int x = pos[0] % c_numCells;
            int y = pos[1] % c_numCells;
            return vectors[y * c_numCells + x];
        };

        MakePerlinNoise("perlin_blue_same_1.png", c_imageSize, c_cellSize, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_blue_same_2.png", c_imageSize, c_cellSize, 2, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_blue_same_3.png", c_imageSize, c_cellSize, 3, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // blue noise, different vectors per octave
    {
        // Load the 16x16 blue noise
        std::vector<Vec2> vectors(c_numCells * c_numCells);
        {
            int w, h, c;
            uint8_t* pixels = stbi_load("BlueNoise16.png", &w, &h, &c, 4);

            if (w != c_numCells || h != c_numCells)
            {
                printf("ERROR! Invalid size blue noise texture!");
                return 1;
            }

            for (size_t index = 0; index < c_numCells * c_numCells; ++index)
            {
                float angle = c_twoPi * float(pixels[index * 4 + 0]) / 255.0f;
                vectors[index] = Vec2
                {
                    cos(angle),
                    sin(angle)
                };
            }

            stbi_image_free(pixels);
        }

        auto UnitVectorAtCell = [&vectors](const IVec2& pos, int octave)
        {
            Vec2 offsetUV = R2(octave);

            int x = (pos[0] + int(offsetUV[0] * float(c_numCells))) % c_numCells;
            int y = (pos[1] + int(offsetUV[0] * float(c_numCells))) % c_numCells;
            return vectors[y * c_numCells + x];
        };

        MakePerlinNoise("perlin_blue_different_1.png", c_imageSize, c_cellSize, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_blue_different_2.png", c_imageSize, c_cellSize, 2, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_blue_different_3.png", c_imageSize, c_cellSize, 3, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // IGN, same vectors per octave
    {
        auto UnitVectorAtCell = [](const IVec2& pos, int octave)
        {
            float angle = IGN(pos[0], pos[1]);

            return Vec2
            {
                cos(angle),
                sin(angle)
            };
        };

        MakePerlinNoise("perlin_ign_same_1.png", c_imageSize, c_cellSize, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_ign_same_2.png", c_imageSize, c_cellSize, 2, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_ign_same_3.png", c_imageSize, c_cellSize, 3, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // IGN, different vectors per octave
    {
        auto UnitVectorAtCell = [](const IVec2& pos, int octave)
        {
            Vec2 offsetUV = R2(octave);

            IVec2 offsetPos;
            offsetPos[0] = pos[0] + int(offsetUV[0] * float(c_numCells));
            offsetPos[1] = pos[1] + int(offsetUV[1] * float(c_numCells));

            float angle = IGN(offsetPos[0], offsetPos[1]);

            return Vec2
            {
                cos(angle),
                sin(angle)
            };
        };

        MakePerlinNoise("perlin_ign_different_1.png", c_imageSize, c_cellSize, 1, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_ign_different_2.png", c_imageSize, c_cellSize, 2, SmoothingMode::Smoother, UnitVectorAtCell);
        MakePerlinNoise("perlin_ign_different_3.png", c_imageSize, c_cellSize, 3, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // larger render using blue 64x64
    {
        // Load the 64x64 blue noise
        std::vector<Vec2> vectors(64 * 64);
        {
            int w, h, c;
            uint8_t* pixels = stbi_load("BlueNoise64.png", &w, &h, &c, 4);

            if (w != 64 || h != 64)
            {
                printf("ERROR! Invalid size blue noise texture!");
                return 1;
            }

            for (size_t index = 0; index < 64 * 64; ++index)
            {
                float angle = c_twoPi * float(pixels[index * 4 + 0]) / 255.0f;
                vectors[index] = Vec2
                {
                    cos(angle),
                    sin(angle)
                };
            }

            stbi_image_free(pixels);
        }

        auto UnitVectorAtCell = [&vectors](const IVec2& pos, int octave)
        {
            Vec2 offsetUV = R2(octave);

            int x = (pos[0] + int(offsetUV[0] * float(64))) % 64;
            int y = (pos[1] + int(offsetUV[0] * float(64))) % 64;
            return vectors[y * 64 + x];
        };

        MakePerlinNoise("perlin_big_blue64x64.png", 2*c_imageSize, 2*c_imageSize / 64, 1, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    // larger render using blue 16x16
    {
        // Load the 16x16 blue noise
        std::vector<Vec2> vectors(16 * 16);
        {
            int w, h, c;
            uint8_t* pixels = stbi_load("BlueNoise16.png", &w, &h, &c, 4);

            if (w != 16 || h != 16)
            {
                printf("ERROR! Invalid size blue noise texture!");
                return 1;
            }

            for (size_t index = 0; index < 16 * 16; ++index)
            {
                float angle = c_twoPi * float(pixels[index * 4 + 0]) / 255.0f;
                vectors[index] = Vec2
                {
                    cos(angle),
                    sin(angle)
                };
            }

            stbi_image_free(pixels);
        }

        auto UnitVectorAtCell = [&vectors](const IVec2& pos, int octave)
        {
            Vec2 offsetUV = R2(octave);

            int x = (pos[0] + int(offsetUV[0] * float(16))) % 16;
            int y = (pos[1] + int(offsetUV[0] * float(16))) % 16;
            return vectors[y * 16 + x];
        };

        MakePerlinNoise("perlin_big_blue16x16.png", 4 * 2 * c_imageSize, 4 * 2 * c_imageSize / 16, 1, SmoothingMode::Smoother, UnitVectorAtCell);
    }

    system("python DFT.py");
}

/*

Eevee's perlin noise tutorial: https://eev.ee/blog/2016/05/29/perlin-noise/
 * Also use the other method she talks about of selecting pre-made vectora.
 * Envies post talks about clumps being bad, so blue seems like a good idea? might need to check out the paper.

*/
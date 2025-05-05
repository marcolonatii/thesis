#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Set default precision for floats and sampler2DArray
precision highp float;
precision highp sampler2DArray;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize;
uniform int uNumMasks;
uniform float uOpacity;
uniform bool uBorder;

#define MAX_MASKS 100
uniform sampler2DArray uMaskTextures;
uniform vec4 uMaskColors[MAX_MASKS];

uniform float uTime;
uniform vec2 uClickPos;
uniform int uActiveMask;

out vec4 fragColor;

/**
 * Reduces the saturation of a color by blending it with its luminance.
 * @param color The input color to desaturate.
 * @param saturationFactor Factor between 0 (grayscale) and 1 (original).
 * @return The desaturated color.
 */
vec4 lowerSaturation(vec4 color, float saturationFactor) {
  float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;
  vec3 gray = vec3(luminance);
  vec3 saturated = mix(gray, color.rgb, saturationFactor);
  return vec4(saturated, color.a);
}

/**
 * Detects edges in a mask texture layer using a Sobel-like operator.
 * @param textureSampler The texture array containing mask layers.
 * @param layer The layer index to sample from.
 * @param coverage The sampling distance for edge detection.
 * @param edgeColor The color to apply to detected edges.
 * @return edgeColor if an edge is detected, else transparent black.
 */
vec4 detectEdges(sampler2DArray textureSampler, float layer, float coverage, vec4 edgeColor) {
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);
  vec2 texOffset = 1.0f / uSize;

  // Sample neighboring pixels from the specified layer using tvTexCoord
  float tLeft = texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(-coverage, coverage), layer)).r;
  float tRight = texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(coverage, -coverage), layer)).r;
  float bLeft = texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(-coverage, -coverage), layer)).r;
  float bRight = texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(coverage, coverage), layer)).r;

  float xEdge = tLeft + 2.0f * texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(-coverage, 0), layer)).r + bLeft 
                - tRight - 2.0f * texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(coverage, 0), layer)).r - bRight;
  float yEdge = tLeft + 2.0f * texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(0, coverage), layer)).r + tRight 
                - bLeft - 2.0f * texture(textureSampler, vec3(tvTexCoord + texOffset * vec2(0, -coverage), layer)).r - bRight;

  float result = sqrt(xEdge * xEdge + yEdge * yEdge);
  return result > 1e-6f ? edgeColor : vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

/**
 * Adjusts texture coordinates based on a bounding box and aspect ratio.
 * @param vTexCoord The original texture coordinates.
 * @param bbox The bounding box in normalized coordinates.
 * @param aspectRatio The aspect ratio of the viewport.
 * @return Adjusted texture coordinates centered and scaled.
 */
vec2 calculateAdjustedTexCoord(vec2 vTexCoord, vec4 bbox, float aspectRatio) {
  vec2 center = vec2((bbox.x + bbox.z) * 0.5f, bbox.w);
  float radiusX = abs(bbox.z - bbox.x);
  float radiusY = radiusX / aspectRatio;
  float scale = 1.0f;
  radiusX *= scale;
  radiusY *= scale;
  vec2 adjustedTexCoord = (vTexCoord - center) / vec2(radiusX, radiusY) + vec2(0.5f);
  return adjustedTexCoord;
}

void main() {
  vec4 color = texture(uSampler, vTexCoord);
  float saturationFactor = 0.7;
  float aspectRatio = uSize.y / uSize.x;
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);

  vec4 finalColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float totalMaskValue = 0.0f;
  vec4 edgeColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float numRipples = 1.75;
  float timeThreshold = 1.1;
  vec2 adjustedClickCoord = calculateAdjustedTexCoord(vTexCoord, vec4(uClickPos, uClickPos + 0.1), aspectRatio);

  for (int i = 0; i < MAX_MASKS; i++) {
    if (i >= uNumMasks) break;

    float maskValue = texture(uMaskTextures, vec3(tvTexCoord, float(i))).r;
    vec4 maskColor = uMaskColors[i];

    maskColor /= 255.0;
    vec4 saturatedColor = lowerSaturation(maskColor, saturationFactor);
    vec4 plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(maskColor.rgb, 0.2);

    if (uActiveMask == i && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor, plainColor, colorFactor));
    }
    if (uTime >= timeThreshold) {
      plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    }

    finalColor += maskValue * plainColor;
    totalMaskValue += maskValue;

    if (edgeColor.a <= 0.0f) {
      edgeColor = detectEdges(uMaskTextures, float(i), 1.25, maskColor);
    }
  }

  if (totalMaskValue > 0.0f) {
    finalColor /= totalMaskValue;
    finalColor = mix(color, finalColor, uOpacity);
  } else {
    finalColor.a = 0.0f;
  }

  if (edgeColor.a > 0.0f && uBorder) {
    finalColor = vec4(vec3(edgeColor), 1.0);
  }
  fragColor = finalColor;
}
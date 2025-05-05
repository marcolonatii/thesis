/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import {hexToRgb} from '@/common/components/video/editor/VideoEditorUtils';
import BaseGLEffect from '@/common/components/video/effects/BaseGLEffect';
import {
  EffectFrameContext,
  EffectInit,
} from '@/common/components/video/effects/Effect';
import vertexShaderSource from '@/common/components/video/effects/shaders/DefaultVert.vert?raw';
import fragmentShaderSource from '@/common/components/video/effects/shaders/Overlay.frag?raw';
import {Tracklet} from '@/common/tracker/Tracker';
import {
  findIndexByTrackletId,
} from '@/common/utils/ShaderUtils';
import {RLEObject, decode} from '@/jscocotools/mask';
import invariant from 'invariant';
import {CanvasForm} from 'pts';
import Logger from '@/common/logger/Logger';

export default class OverlayEffect extends BaseGLEffect {
  private _numMasks: number = 0;
  private _numMasksUniformLocation: WebGLUniformLocation | null = null;
  private _maskTexturesLocation: WebGLUniformLocation | null = null;
  private _maskColorsLocation: WebGLUniformLocation | null = null;
  private _maskTextureArray: WebGLTexture | null = null;
  private _clickPosition: number[] | null = null;
  private _activeMask: number = 0;

  // Maximum number of masks supported, matching the shader's MAX_MASKS
  private static readonly MAX_MASKS: number = 100;

  constructor() {
    super(8);
    this.vertexShaderSource = vertexShaderSource;
    this.fragmentShaderSource = fragmentShaderSource;
  }

  protected setupUniforms(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    init: EffectInit,
  ): void {
    super.setupUniforms(gl, program, init);

    this._numMasksUniformLocation = gl.getUniformLocation(program, 'uNumMasks');
    this._maskTexturesLocation = gl.getUniformLocation(program, 'uMaskTextures');
    this._maskColorsLocation = gl.getUniformLocation(program, 'uMaskColors[0]');

    gl.uniform1i(this._numMasksUniformLocation, this._numMasks);

    this._maskTextureArray = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this._maskTextureArray);
    gl.texImage3D(
      gl.TEXTURE_2D_ARRAY,
      0,
      gl.LUMINANCE,
      init.height, // Swap to height
      init.width,  // Swap to width
      OverlayEffect.MAX_MASKS,
      0,
      gl.LUMINANCE,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.uniform1i(this._maskTexturesLocation, 1);

    gl.uniform2f(gl.getUniformLocation(program, 'uSize'), init.width, init.height);
  }

  apply(form: CanvasForm, context: EffectFrameContext, _tracklets: Tracklet[]) {
    const gl = this._gl;
    const program = this._program;

    invariant(gl !== null, 'WebGL2 context is required');
    invariant(program !== null, 'No WebGL program found');

    Logger.debug(`Applying effect: masks.length=${context.masks.length}, maskColors.length=${context.maskColors?.length || 0}, tracklets.length=${_tracklets.length}`);

    if (context.masks.length === 0) {
      Logger.warn('No masks provided in context. Skipping effect application.');
      const ctx = form.ctx;
      invariant(this._canvas !== null, 'Canvas is required');
      ctx.drawImage(context.frame, 0, 0);
      return;
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const opacity = [0.5, 0.75, 0.35, 0.95][this.variant % 4];
    gl.uniform1f(
      gl.getUniformLocation(program, 'uTime'),
      context.timeParameter ?? 1.5,
    );
    gl.uniform1f(gl.getUniformLocation(program, 'uOpacity'), opacity);
    gl.uniform1i(
      gl.getUniformLocation(program, 'uBorder'),
      this.variant % this.numVariants < 4 ? 1 : 0,
    );

    if (context.actionPoint) {
      const clickPos = [
        context.actionPoint.position[0] / context.width,
        context.actionPoint.position[1] / context.height,
      ];
      this._clickPosition = clickPos;
      this._activeMask = findIndexByTrackletId(
        context.actionPoint.objectId,
        _tracklets,
      );
    }

    gl.uniform2fv(
      gl.getUniformLocation(program, 'uClickPos'),
      this._clickPosition ?? [0, 0],
    );
    gl.uniform1i(
      gl.getUniformLocation(program, 'uActiveMask'),
      this._activeMask,
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._frameTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      context.width,
      context.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      context.frame,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const numMasks = Math.min(context.masks.length, OverlayEffect.MAX_MASKS);
    gl.uniform1i(this._numMasksUniformLocation, numMasks);

    if (numMasks < context.masks.length) {
      Logger.warn(`Mask count ${context.masks.length} exceeds MAX_MASKS ${OverlayEffect.MAX_MASKS}. Truncating to ${numMasks}.`);
    }

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this._maskTextureArray);
    for (let index = 0; index < numMasks; index++) {
      const mask = context.masks[index];
      const decodedMask = decode([mask.bitmap as RLEObject]);
      const maskData = decodedMask.data as Uint8Array;
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texSubImage3D(
        gl.TEXTURE_2D_ARRAY,
        0,
        0,
        0,
        index,
        context.height, // Swap to height
        context.width,  // Swap to width
        1,
        gl.LUMINANCE,
        gl.UNSIGNED_BYTE,
        maskData,
      );
    }

    if (!Array.isArray(context.maskColors) || context.maskColors.length < numMasks) {
      Logger.warn(`Invalid or insufficient maskColors: ${JSON.stringify(context.maskColors)}. Using fallback colors.`);
    }

    const colorArray = new Float32Array(4 * numMasks);
    for (let index = 0; index < numMasks; index++) {
      let color;
      try {
        const hexColor = context.maskColors && context.maskColors[index] ? context.maskColors[index] : '#808080';
        color = hexToRgb(hexColor);
        if (!color || typeof color.r !== 'number') {
          throw new Error(`Invalid color from hexToRgb for mask ${index}`);
        }
      } catch (e) {
        Logger.error(`Failed to parse color for mask ${index}: ${e}. Using fallback.`);
        color = { r: 128, g: 128, b: 128, a: 255 };
      }
      colorArray[index * 4 + 0] = color.r / 255.0;
      colorArray[index * 4 + 1] = color.g / 255.0;
      colorArray[index * 4 + 2] = color.b / 255.0;
      colorArray[index * 4 + 3] = color.a / 255.0;
    }

    if (!(colorArray instanceof Float32Array) || colorArray.length === 0) {
      Logger.error('colorArray is invalid or empty. This should not happen with valid numMasks.');
    } else if (this._maskColorsLocation !== null) {
      gl.uniform4fv(this._maskColorsLocation, colorArray);
    } else {
      Logger.error('maskColors uniform location is null. Cannot set colors.');
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    const ctx = form.ctx;
    invariant(this._canvas !== null, 'Canvas is required');
    ctx.drawImage(this._canvas, 0, 0);
    this._clickPosition = null;
  }

  async cleanup(): Promise<void> {
    super.cleanup();

    if (this._gl != null && this._maskTextureArray != null) {
      this._gl.deleteTexture(this._maskTextureArray);
      this._maskTextureArray = null;
    }
  }
}
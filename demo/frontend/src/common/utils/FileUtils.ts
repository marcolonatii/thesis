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
import Logger from '@/common/logger/Logger';

type Range = {
  start: number;
  end: number;
};

type FileStreamPart = {
  data: Uint8Array;
  range: Range;
  contentLength: number;
};

export type ImageFrameInfo = {
  url: string;
  index: number;
};

export type FileStream = AsyncGenerator<FileStreamPart, File | null, null>;
export type ImageFrameStream = AsyncGenerator<ImageFrameInfo[], string | null, null>;

/**
 * Asynchronously generates a SHA-256 hash for a Blob object.
 *
 * DO NOT USE this function casually. Computing the SHA-256 is expensive and can
 * take several 100 milliseconds to complete.
 *
 * @param blob - The Blob object to be hashed.
 * @returns A Promise that resolves to a string representing the SHA-256 hash of
 * the Blob.
 */
export async function hashBlob(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  // Crypto subtle is only availabe in secure contexts. For example, this will
  // be the case when running the project locally with http protocol.
  // https://developer.mozilla.org/en-US/docs/Web/API/Crypto/subtle
  if (crypto.subtle != null) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }
  // If not secure context, return random string
  return (Math.random() + 1).toString(36).substring(7);
}

/**
 * Legacy function for streaming MP4 files
 */
export async function* streamFile(url: string, init?: RequestInit): FileStream {
  try {
    const response = await fetch(url, init);

    let blob: Blob;

    // Try to download the file with a stream reader. This has the benefit
    // of providing progress during the download. It requires the body and
    // Content-Length. As a fallback, it uses the blob function on the
    // response object.
    const contentLength = response.headers.get('Content-Length');
    if (response.body != null && contentLength != null) {
      const totalLength = parseInt(contentLength);
      const chunks: Uint8Array[] = [];
      let start = 0;
      let end = 0;

      const reader = response.body.getReader();
      try {
        while (true) {
          const {done, value} = await reader.read();
          if (done) {
            break;
          }

          start = end;
          end += value.length;

          yield {
            data: value,
            range: {start, end},
            contentLength: totalLength,
          };
        }
      } finally {
        reader.releaseLock();
      }
      blob = new Blob(chunks);
    } else {
      blob = await response.blob();
    }

    const filename = await hashBlob(blob);
    return new File([blob], `${filename}.mp4`);
  } catch (error) {
    Logger.error('aborting download due to component unmount', error);
  }
  return null;
}

/**
 * Fetches frame images from a directory path
 * @param directoryPath The path to the directory containing image frames
 * @param init Optional request initialization
 */
export async function* streamImageFrames(directoryPath: string, init?: RequestInit): ImageFrameStream {
  try {
    // First, fetch the directory listing to get all image files
    const response = await fetch(`${directoryPath}`, {
      ...init,
      headers: {
        ...init?.headers,
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      Logger.error(`Failed to fetch directory listing: ${response.statusText}`);
      return null;
    }
    
    const data = await response.json();
    if (!Array.isArray(data)) {
      Logger.error('Expected directory listing to be an array');
      return null;
    }
    
    // Filter for image files and sort them
    const imageFrames = data
      .filter((item: any) => {
        const filename = typeof item === 'string' ? item : item.name;
        return /\.(jpe?g|png)$/i.test(filename);
      })
      .sort((a: any, b: any) => {
        const nameA = typeof a === 'string' ? a : a.name;
        const nameB = typeof b === 'string' ? b : b.name;
        
        // Try to sort numerically if frames are named with numbers
        const numA = parseInt(nameA.match(/\d+/)?.[0] || '0');
        const numB = parseInt(nameB.match(/\d+/)?.[0] || '0');
        
        if (!isNaN(numA) && !isNaN(numB)) {
          return numA - numB;
        }
        
        // Fall back to alphabetical sorting
        return nameA.localeCompare(nameB);
      })
      .map((item: any, index: number) => {
        const filename = typeof item === 'string' ? item : item.name;
        return {
          url: `${directoryPath}/${filename}`,
          index
        };
      });
    
    // Yield all frames at once
    yield imageFrames;
    
    return directoryPath;
  } catch (error) {
    Logger.error('Error streaming image frames', error);
    return null;
  }
}

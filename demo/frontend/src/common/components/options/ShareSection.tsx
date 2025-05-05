/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import DownloadOption from './DownloadOption'; // Existing rendered video download
import DownloadMasksOption from './DownloadMasksOption'; // New
import DownloadImagesOption from './DownloadImagesOption'; // New
import DownloadYoloLabelsOption from './DownloadYoloLabelsOption'; // New (for Boxes)
import DownloadYoloFormatOption from './DownloadYoloFormatOption'; // New (for All)

export default function ShareSection() {
  return (
    // Add some spacing between buttons using gap
    <div className="p-5 md:p-8 flex flex-col gap-4">
        {/* Existing Download (Rendered Video) */}
        <DownloadOption />

        {/* New Download Options */}
        <DownloadMasksOption />
        <DownloadYoloLabelsOption /> {/* "Download Boxes" */}
        <DownloadImagesOption />
        <DownloadYoloFormatOption /> {/* "Download All" */}
    </div>
  );
}